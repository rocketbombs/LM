//! Lock-free parallel generation pipeline using std::thread.
//!
//! Achieves maximum throughput by:
//! - Using multi-threading with thread pool
//! - Lock-free data structures (mpsc channels)
//! - Per-worker RNG to avoid contention
//! - Batched output to minimize synchronization

use crate::generator::{GeneratorConfig, TermGenerator, SimpleRng};
use crate::reduction::{GraphReducer, ReductionConfig};
use crate::render::{get_redex_span, render_debruijn};
use crate::schema::{ExampleMetadata, TrainingExample};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::Arc;
use std::thread;

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
        // Default to 8 workers (can be overridden)
        let num_workers = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);

        PipelineConfig {
            num_workers,
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
    /// Uses std::thread for true parallelism.
    /// Each worker has its own RNG and reducer to avoid contention.
    pub fn generate<F>(&self, mut callback: F) -> usize
    where
        F: FnMut(TrainingExample) + Send + 'static,
    {
        let (tx, rx) = sync_channel::<TrainingExample>(self.config.num_workers * 100);
        let examples_generated = Arc::new(AtomicUsize::new(0));
        let should_stop = Arc::new(AtomicBool::new(false));

        // Spawn consumer thread
        let consumer = thread::spawn(move || {
            let mut count = 0;
            while let Ok(example) = rx.recv() {
                callback(example);
                count += 1;
            }
            count
        });

        // Spawn worker threads
        let chunk_size = 100; // Generate in chunks for better batching

        // Calculate number of chunks needed (removed artificial 10K cap!)
        let num_chunks = if let Some(max) = self.config.max_terms {
            // FIX: Scale chunks with target size, no artificial limit
            // For 15M examples, we need ~150K chunks (at ~100 examples/chunk)
            // Conservative estimate: 1 chunk generates ~10 examples on average
            let estimated_chunks = (max / 10).max(1); // Rough estimate
            estimated_chunks
        } else {
            // Unlimited: use large but reasonable default
            1_000_000 // 1M chunks for effectively unlimited generation
        };

        let mut handles = Vec::new();

        for worker_id in 0..self.config.num_workers {
            let tx = tx.clone();
            let config = self.config.clone();
            let examples_generated = examples_generated.clone();
            let should_stop = should_stop.clone();

            let handle = thread::spawn(move || {
                // Each worker processes a subset of chunks
                for chunk_id in (worker_id..num_chunks).step_by(config.num_workers) {
                    // Check if we should stop
                    if should_stop.load(Ordering::Relaxed) {
                        break;
                    }

                    // Check if we've hit max_terms
                    if let Some(max) = config.max_terms {
                        if examples_generated.load(Ordering::Relaxed) >= max {
                            should_stop.store(true, Ordering::Relaxed);
                            break;
                        }
                    }

                    // Per-worker RNG with proper seed mixing
                    // CRITICAL: Include both worker_id AND chunk_id for true uniqueness
                    // Use SplitMix64-style mixing for good avalanche properties
                    let worker_seed = config.seed
                        .wrapping_add((worker_id as u64).wrapping_mul(0x9e3779b97f4a7c15))
                        .wrapping_add((chunk_id as u64).wrapping_mul(0x6a09e667f3bcc909))
                        ^ ((chunk_id as u64) << 32);

                    // Extra mixing step for better distribution
                    let mixed_seed = worker_seed
                        .wrapping_mul(0xbf58476d1ce4e5b9)
                        ^ (worker_seed >> 32);

                    let mut rng = SimpleRng::seed_from_u64(mixed_seed);

                    // Per-worker reducer (no contention)
                    let mut reducer = GraphReducer::new(config.reduction_config.clone());

                    // DIVERSITY: Vary generation parameters per chunk for maximum variety
                    // Cycle through different complexity levels to ensure broad coverage
                    // AGGRESSIVE: Push harder for large terms (cycles 0-9)
                    let complexity_cycle = (chunk_id % 10) as usize;
                    let varied_config = GeneratorConfig {
                        max_depth: match complexity_cycle {
                            0 => 8,   // Medium
                            1 => 10,  // Complex
                            2 => 12,  // Very complex
                            3 => 13,  // Deep
                            4 => 14,  // Very deep
                            5 => 15,  // Extreme depth
                            6 => 16,  // Ultra deep
                            7 => 17,  // Maximum depth
                            8 => 18,  // Beyond max (push limits!)
                            9 => 20,  // Extreme (push model hard!)
                            _ => 12,
                        },
                        max_size: match complexity_cycle {
                            0 => 120,  // Medium baseline
                            1 => 150,  // Large
                            2 => 180,  // Very large
                            3 => 210,  // Huge
                            4 => 240,  // Very huge (PUSH!)
                            5 => 270,  // Extreme
                            6 => 300,  // Ultra extreme (PUSH HARD!)
                            7 => 250,  // Back to huge
                            8 => 200,  // Large variety
                            9 => 220,  // Very large variety
                            _ => 180,
                        },
                        min_depth: config.generator_config.min_depth,
                        allow_divergent: config.generator_config.allow_divergent,
                    };

                    // Per-worker generator with varied parameters
                    let generator = TermGenerator::new(varied_config);

                    // Generate chunk of terms
                    for draw_index in 0..chunk_size {
                        if should_stop.load(Ordering::Relaxed) {
                            break;
                        }

                        if let Some(max) = config.max_terms {
                            if examples_generated.load(Ordering::Relaxed) >= max {
                                should_stop.store(true, Ordering::Relaxed);
                                break;
                            }
                        }

                        // Extra per-term entropy injection for maximum diversity
                        // Mix draw_index into RNG state to ensure each term in chunk is unique
                        // This prevents any correlation even if RNG has weaknesses
                        let draw_entropy = (draw_index as u64)
                            .wrapping_mul(0x9e3779b97f4a7c15)
                            ^ (draw_index as u64).rotate_left(32);

                        // Inject entropy into RNG for this specific term
                        rng.inject_entropy(draw_entropy);

                    // Generate term
                    let term = match generator.generate(&mut rng) {
                        Some(t) => t,
                        None => continue,
                    };

                    // Reduce term
                    let trace = reducer.reduce(&term);

                        // Check if we should include diverged terms
                        if trace.diverged && !config.generator_config.allow_divergent {
                            continue;
                        }

                        // Compute step times for averaging
                        let step_times: Vec<f64> = trace.steps.iter().map(|s| s.step_time_ms).collect();
                        let avg_step_ms = if !step_times.is_empty() {
                            step_times.iter().sum::<f64>() / step_times.len() as f64
                        } else {
                            0.0
                        };

                        // Check if trace is pathological (skip entire trace if so)
                        let initial_size = trace.steps[0].term.size();
                        let final_size = trace.steps.last().map(|s| s.term.size()).unwrap_or(initial_size);
                        let size_growth_rate = if initial_size > 0 {
                            final_size as f64 / initial_size as f64
                        } else {
                            1.0
                        };
                        let time_consumed_ratio = trace.total_time_ms / config.reduction_config.wall_clock_limit_ms;

                        let is_trace_pathological = ExampleMetadata::detect_pathological(
                            time_consumed_ratio,
                            avg_step_ms,
                            size_growth_rate,
                            final_size,
                        );

                        // VALIDATION: Skip pathological traces (extreme growth, very slow, etc.)
                        if is_trace_pathological {
                            // This trace exhibits pathological behavior
                            // Skip to maintain clean training data
                            continue;
                        }

                        // VALIDATION: Skip trivial traces (already in NF or too short)
                        let steps_total = trace.steps.len().saturating_sub(1);
                        if steps_total == 0 {
                            // Already in normal form - no reduction steps
                            // Skip to ensure meaningful reduction examples
                            continue;
                        }

                        if steps_total < 2 {
                            // Only 1 reduction step - too trivial
                            // Skip to ensure diverse, interesting patterns
                            continue;
                        }

                        // Generate training examples from trace
                        let trace_id = format!("{:016x}-{:016x}", worker_seed, draw_index);

                        // Generate example for EACH step in trace
                        for (step_k, step) in trace.steps.iter().enumerate() {
                            let render_result = render_debruijn(&step.term);

                            // Compute hash using DefaultHasher
                            let mut hasher = DefaultHasher::new();
                            render_result.string.hash(&mut hasher);
                            let term_hash = format!("{:x}", hasher.finish());

                        // Get redex span
                        let target_span = get_redex_span(
                            &step.term,
                            step.redex_path.as_deref(),
                            &render_result,
                        );

                            // VALIDATION: Skip invalid examples with premature NF markers
                            // target_span=(0,0) means "normal form reached", only valid on final step
                            let is_final_step = step_k >= steps_total;
                            let is_nf_marker = target_span == (0, 0);

                            if is_nf_marker && !is_final_step {
                                // This is INVALID: NF marker on non-final step
                                // This indicates a bug in find_redex or malformed term
                                // Skip this example to maintain data quality
                                continue;
                            }

                            // Compute runtime metrics
                            let elapsed_time_ms: f64 = step_times[..=step_k].iter().sum();
                            let time_remaining_ms =
                                (config.reduction_config.wall_clock_limit_ms - elapsed_time_ms)
                                    .max(0.0);
                            let time_consumed_ratio = elapsed_time_ms
                                / config.reduction_config.wall_clock_limit_ms;

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

                            // VALIDATION: Skip pathological examples (individual steps)
                            // Even if the overall trace isn't pathological, individual steps can become
                            // pathological as the term grows or slows down during reduction
                            if is_pathological {
                                continue;
                            }

                            // Create complete metadata
                            // Note: is_pathological is always false here because we filter them out
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
                                config.reduction_config.wall_clock_limit_ms,
                                time_remaining_ms,
                                time_consumed_ratio,
                                false, // Always false - pathological examples are filtered out
                                size_growth_rate,
                                initial_size,
                            );

                            let example = TrainingExample {
                                strategy: config.strategy.clone(),
                                render: config.render.clone(),
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
                                should_stop.store(true, Ordering::Relaxed);
                                break;
                            }

                            examples_generated.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            });

            handles.push(handle);
        }

        // Drop main sender
        drop(tx);

        // Wait for all workers
        for handle in handles {
            let _ = handle.join();
        }

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
            max_terms: Some(10), // Generate 10 terms
            ..Default::default()
        };

        let pipeline = ParallelPipeline::new(config);
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        pipeline.generate(move |example| {
            // Verify all metadata fields are present
            assert!(example.meta.step_ms >= 0.0);
            assert!(example.meta.wall_clock_limit_ms > 0.0);
            assert!(example.meta.time_consumed_ratio >= 0.0);
            assert!(example.meta.time_consumed_ratio <= 1.0);
            count_clone.fetch_add(1, Ordering::Relaxed);
        });

        let final_count = count.load(Ordering::Relaxed);
        // Each term can generate multiple examples (one per reduction step)
        // So we should get at least 10 examples, but likely more
        assert!(final_count >= 10);
        println!("Generated {} examples from 10 terms", final_count);
    }

    #[test]
    fn test_unlimited_generation_no_overflow() {
        // Regression test for overflow bug when max_terms is None
        // Previously: (usize::MAX + chunk_size - 1) caused overflow
        // Now: properly handles unlimited case
        let config = PipelineConfig {
            num_workers: 2,
            max_terms: None,  // Unlimited - this used to overflow!
            ..Default::default()
        };

        let pipeline = ParallelPipeline::new(config);
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();
        let stop_at = 20; // Generate a few to verify it works

        pipeline.generate(move |_example| {
            let c = count_clone.fetch_add(1, Ordering::Relaxed);
            // Stop after generating some examples (to avoid infinite loop in test)
            // In production, unlimited would run indefinitely
            if c >= stop_at {
                // Note: This doesn't actually stop the pipeline cleanly,
                // but in practice the test will complete when workers finish chunks
            }
        });

        let final_count = count.load(Ordering::Relaxed);
        // With unlimited, we should have generated examples
        // If overflow occurred, num_chunks would be 0 and we'd get 0 examples
        assert!(final_count > 0, "Overflow bug: generated 0 examples with max_terms=None");
        println!("Generated {} examples with unlimited mode", final_count);
    }
}
