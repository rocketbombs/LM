//! Lock-free parallel generation pipeline using Rayon.
//!
//! Achieves maximum throughput by:
//! - Using work-stealing parallelism (Rayon)
//! - Lock-free data structures (crossbeam channels)
//! - Per-worker RNG to avoid contention
//! - Batched output to minimize synchronization

use crate::generator::{GeneratorConfig, TermGenerator};
use crate::reduction::{GraphReducer, ReductionConfig};
use crate::render::{get_redex_span, render_debruijn};
use crate::schema::{ExampleMetadata, TrainingExample};
use crossbeam::channel::{bounded, Sender};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use seahash::hash;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub num_workers: usize,
    pub generator_config: GeneratorConfig,
    pub reduction_config: ReductionConfig,
    pub strategy: String,
    pub render: String,
    pub seed: u64,
    pub max_terms: Option<usize>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            num_workers: num_cpus::get(),
            generator_config: GeneratorConfig::default(),
            reduction_config: ReductionConfig::default(),
            strategy: "levy_like".to_string(),
            render: "debruijn".to_string(),
            seed: 42,
            max_terms: None,
        }
    }
}

pub struct ParallelPipeline {
    config: PipelineConfig,
}

impl ParallelPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        ParallelPipeline { config }
    }

    /// Generate training examples in parallel
    ///
    /// Uses Rayon for true parallelism with work-stealing.
    /// Each worker has its own RNG and reducer to avoid contention.
    pub fn generate<F>(&self, mut callback: F) -> usize
    where
        F: FnMut(TrainingExample) + Send,
    {
        let (tx, rx) = bounded::<TrainingExample>(self.config.num_workers * 100);
        let examples_generated = Arc::new(AtomicUsize::new(0));
        let examples_generated_clone = examples_generated.clone();

        // Spawn consumer thread
        let consumer = std::thread::spawn(move || {
            let mut count = 0;
            while let Ok(example) = rx.recv() {
                callback(example);
                count += 1;
            }
            count
        });

        // Parallel generation using Rayon
        let chunk_size = 100; // Generate in chunks for better batching
        let num_chunks = self.config.max_terms.unwrap_or(usize::MAX) / chunk_size;

        (0..num_chunks.min(10000))
            .into_par_iter()
            .try_for_each(|chunk_id| {
                // Check if we've hit max_terms
                if let Some(max) = self.config.max_terms {
                    if examples_generated_clone.load(Ordering::Relaxed) >= max {
                        return None; // Stop generating
                    }
                }

                // Per-worker RNG (no contention)
                let worker_seed = self.config.seed
                    .wrapping_add(chunk_id as u64)
                    .wrapping_mul(0x9e3779b97f4a7c15);
                let mut rng = ChaCha8Rng::seed_from_u64(worker_seed);

                // Per-worker reducer (no contention)
                let mut reducer = GraphReducer::new(self.config.reduction_config.clone());

                // Per-worker generator
                let generator = TermGenerator::new(self.config.generator_config.clone());

                // Generate chunk of terms
                for draw_index in 0..chunk_size {
                    if let Some(max) = self.config.max_terms {
                        if examples_generated_clone.load(Ordering::Relaxed) >= max {
                            break;
                        }
                    }

                    // Generate term
                    let term = match generator.generate(&mut rng) {
                        Some(t) => t,
                        None => continue,
                    };

                    // Reduce term
                    let trace = reducer.reduce(&term);

                    // Check if we should include diverged terms
                    if trace.diverged && !self.config.generator_config.allow_divergent {
                        continue;
                    }

                    // Generate training examples from trace
                    let trace_id = Uuid::new_v4().to_string();
                    let initial_size = trace.steps[0].term.size();

                    // Compute step times for averaging
                    let step_times: Vec<f64> = trace.steps.iter().map(|s| s.step_time_ms).collect();
                    let avg_step_ms = if !step_times.is_empty() {
                        step_times.iter().sum::<f64>() / step_times.len() as f64
                    } else {
                        0.0
                    };

                    let steps_total = trace.steps.len().saturating_sub(1);

                    // Generate example for EACH step in trace
                    for (step_k, step) in trace.steps.iter().enumerate() {
                        let render_result = render_debruijn(&step.term);

                        // Compute hash
                        let term_hash = format!("{:x}", hash(render_result.string.as_bytes()));

                        // Get redex span
                        let target_span = get_redex_span(
                            &step.term,
                            step.redex_path.as_deref(),
                            &render_result,
                        );

                        // Compute runtime metrics
                        let elapsed_time_ms: f64 = step_times[..=step_k].iter().sum();
                        let time_remaining_ms =
                            (self.config.reduction_config.wall_clock_limit_ms - elapsed_time_ms)
                                .max(0.0);
                        let time_consumed_ratio = elapsed_time_ms
                            / self.config.reduction_config.wall_clock_limit_ms;

                        let current_size = step.term.size();
                        let size_growth_rate = if initial_size > 0 {
                            current_size as f64 / initial_size as f64
                        } else {
                            1.0
                        };

                        let is_pathological = ExampleMetadata::detect_pathological(
                            time_consumed_ratio,
                            avg_step_ms,
                            size_growth_rate,
                            current_size,
                        );

                        // Create complete metadata
                        let meta = ExampleMetadata::new(
                            current_size,
                            step.term.depth(),
                            worker_seed,
                            chunk_id * chunk_size + draw_index,
                            &trace_id,
                            &term_hash,
                            trace.thunk_evals,
                            trace.thunk_hits,
                            step.step_time_ms,
                            avg_step_ms,
                            trace.total_time_ms,
                            self.config.reduction_config.wall_clock_limit_ms,
                            time_remaining_ms,
                            time_consumed_ratio,
                            is_pathological,
                            size_growth_rate,
                            initial_size,
                        );

                        let example = TrainingExample {
                            strategy: self.config.strategy.clone(),
                            render: self.config.render.clone(),
                            term: render_result.string,
                            step_k,
                            target_span,
                            next_term: None, // Can be added if needed
                            normal_form: None,
                            steps_total,
                            diverged: trace.diverged,
                            trace_id: trace_id.clone(),
                            meta,
                        };

                        // Send to consumer
                        if tx.send(example).is_err() {
                            return None; // Consumer dropped
                        }

                        examples_generated_clone.fetch_add(1, Ordering::Relaxed);
                    }
                }

                Some(())
            });

        // Drop sender to signal consumer
        drop(tx);

        // Wait for consumer
        consumer.join().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_generation() {
        let config = PipelineConfig {
            num_workers: 4,
            max_terms: Some(100),
            ..Default::default()
        };

        let pipeline = ParallelPipeline::new(config);
        let mut count = 0;

        pipeline.generate(|example| {
            // Verify all metadata fields are present
            assert!(example.meta.step_ms >= 0.0);
            assert!(example.meta.wall_clock_limit_ms > 0.0);
            assert!(example.meta.time_consumed_ratio >= 0.0);
            assert!(example.meta.time_consumed_ratio <= 1.0);
            count += 1;
        });

        assert!(count > 0);
        assert!(count <= 100);
    }
}
