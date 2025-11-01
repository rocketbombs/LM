//! High-performance lambda calculus inference engine.
//!
//! Compares neural model reduction against classical normal-order reduction
//! with extreme throughput optimization.
//!
//! Usage:
//!   lambda-infer --model model.onnx --terms 10000 --workers 16

use clap::Parser;
use lambda_gen_rs::{ClassicalReducer, NeuralReducer, Term, TermGenerator, GeneratorConfig};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "lambda-infer")]
#[command(about = "High-performance lambda calculus inference engine", long_about = None)]
struct Args {
    /// Path to ONNX model file
    #[arg(long)]
    model: Option<String>,

    /// Number of terms to evaluate
    #[arg(long, default_value = "1000")]
    terms: usize,

    /// Number of parallel workers
    #[arg(long, default_value = "16")]
    workers: usize,

    /// Maximum reduction steps per term
    #[arg(long, default_value = "1000")]
    max_steps: usize,

    /// Maximum term depth
    #[arg(long, default_value = "10")]
    max_depth: u32,

    /// Maximum term size
    #[arg(long, default_value = "100")]
    max_size: u32,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Output JSON results file
    #[arg(long)]
    output: Option<String>,

    /// Run classical reducer only (skip neural)
    #[arg(long)]
    classical_only: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkResult {
    reducer_type: String,
    total_terms: usize,
    total_steps: usize,
    total_time_ms: f64,
    avg_time_per_term_ms: f64,
    avg_steps_per_term: f64,
    throughput_terms_per_sec: f64,
    throughput_steps_per_sec: f64,
    convergence_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ComparisonResult {
    classical: BenchmarkResult,
    neural: Option<BenchmarkResult>,
    speedup: Option<f64>,
    step_efficiency: Option<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Set up rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build_global()
        .unwrap();

    println!("Lambda Calculus Inference Engine");
    println!("=================================");
    println!("Workers: {}", args.workers);
    println!("Terms: {}", args.terms);
    println!("Max steps: {}", args.max_steps);
    println!();

    // Generate test terms
    println!("Generating {} test terms...", args.terms);
    let terms = generate_terms(&args)?;
    println!("Generated {} terms\n", terms.len());

    // Run classical reducer
    println!("Running classical normal-order reducer...");
    let classical_result = benchmark_classical(&terms, args.max_steps)?;
    print_result(&classical_result);

    // Run neural reducer if model provided
    let neural_result = if !args.classical_only {
        if let Some(ref model_path) = args.model {
            println!("\nRunning neural reducer...");
            match benchmark_neural(&terms, model_path, args.max_steps) {
                Ok(result) => {
                    print_result(&result);
                    Some(result)
                }
                Err(e) => {
                    eprintln!("Neural reducer failed: {}", e);
                    None
                }
            }
        } else {
            println!("\nSkipping neural reducer (no model specified)");
            None
        }
    } else {
        None
    };

    // Print comparison
    if let Some(ref neural) = neural_result {
        println!("\n=== COMPARISON ===");
        let speedup = classical_result.avg_time_per_term_ms / neural.avg_time_per_term_ms;
        let step_efficiency = neural.avg_steps_per_term / classical_result.avg_steps_per_term;

        println!("Speedup: {:.2}x", speedup);
        println!("Step efficiency: {:.2}%", step_efficiency * 100.0);
        println!(
            "Neural uses {:.1}% of classical steps",
            step_efficiency * 100.0
        );
    }

    // Save results if requested
    if let Some(output_path) = args.output {
        let comparison = ComparisonResult {
            classical: classical_result,
            neural: neural_result.clone(),
            speedup: neural_result.as_ref().map(|n| {
                classical_result.avg_time_per_term_ms / n.avg_time_per_term_ms
            }),
            step_efficiency: neural_result.as_ref().map(|n| {
                n.avg_steps_per_term / classical_result.avg_steps_per_term
            }),
        };

        let json = serde_json::to_string_pretty(&comparison)?;
        std::fs::write(&output_path, json)?;
        println!("\nResults saved to: {}", output_path);
    }

    Ok(())
}

fn generate_terms(args: &Args) -> Result<Vec<Term>, Box<dyn std::error::Error>> {
    let config = GeneratorConfig {
        max_depth: args.max_depth,
        min_depth: 2,
        max_size: args.max_size,
        seed: args.seed,
    };

    let mut generator = TermGenerator::new(config);
    let mut terms = Vec::with_capacity(args.terms);

    for _ in 0..args.terms * 2 {
        // Generate more than needed, filter valid
        if let Some(term) = generator.generate() {
            terms.push(term);
            if terms.len() >= args.terms {
                break;
            }
        }
    }

    Ok(terms)
}

fn benchmark_classical(
    terms: &[Term],
    max_steps: usize,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let total_steps = Arc::new(AtomicUsize::new(0));
    let converged_count = Arc::new(AtomicUsize::new(0));

    // Parallel reduction
    terms.par_iter().for_each(|term| {
        let reducer = ClassicalReducer::new(max_steps);
        let result = reducer.reduce(term);

        total_steps.fetch_add(result.steps, Ordering::Relaxed);
        if result.converged {
            converged_count.fetch_add(1, Ordering::Relaxed);
        }
    });

    let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    let steps = total_steps.load(Ordering::Relaxed);
    let converged = converged_count.load(Ordering::Relaxed);

    Ok(BenchmarkResult {
        reducer_type: "Classical Normal-Order".to_string(),
        total_terms: terms.len(),
        total_steps: steps,
        total_time_ms,
        avg_time_per_term_ms: total_time_ms / terms.len() as f64,
        avg_steps_per_term: steps as f64 / terms.len() as f64,
        throughput_terms_per_sec: terms.len() as f64 / (total_time_ms / 1000.0),
        throughput_steps_per_sec: steps as f64 / (total_time_ms / 1000.0),
        convergence_rate: converged as f64 / terms.len() as f64,
    })
}

fn benchmark_neural(
    terms: &[Term],
    model_path: &str,
    max_steps: usize,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    // Load model once
    let reducer = Arc::new(NeuralReducer::new(model_path, max_steps)?);

    let start = Instant::now();
    let total_steps = Arc::new(AtomicUsize::new(0));
    let total_inference = Arc::new(AtomicUsize::new(0));
    let converged_count = Arc::new(AtomicUsize::new(0));

    // Parallel reduction
    terms.par_iter().for_each(|term| {
        let result = reducer.reduce(term);

        total_steps.fetch_add(result.steps, Ordering::Relaxed);
        total_inference.fetch_add(result.total_inference_ms as usize, Ordering::Relaxed);
        if result.converged {
            converged_count.fetch_add(1, Ordering::Relaxed);
        }
    });

    let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    let steps = total_steps.load(Ordering::Relaxed);
    let converged = converged_count.load(Ordering::Relaxed);
    let inference_ms = total_inference.load(Ordering::Relaxed) as f64;

    println!(
        "  (Inference time: {:.2}ms, {:.1}% of total)",
        inference_ms,
        (inference_ms / total_time_ms) * 100.0
    );

    Ok(BenchmarkResult {
        reducer_type: "Neural".to_string(),
        total_terms: terms.len(),
        total_steps: steps,
        total_time_ms,
        avg_time_per_term_ms: total_time_ms / terms.len() as f64,
        avg_steps_per_term: steps as f64 / terms.len() as f64,
        throughput_terms_per_sec: terms.len() as f64 / (total_time_ms / 1000.0),
        throughput_steps_per_sec: steps as f64 / (total_time_ms / 1000.0),
        convergence_rate: converged as f64 / terms.len() as f64,
    })
}

fn print_result(result: &BenchmarkResult) {
    println!("  Total time: {:.2}ms", result.total_time_ms);
    println!("  Avg time/term: {:.4}ms", result.avg_time_per_term_ms);
    println!("  Avg steps/term: {:.2}", result.avg_steps_per_term);
    println!(
        "  Throughput: {:.0} terms/s, {:.0} steps/s",
        result.throughput_terms_per_sec, result.throughput_steps_per_sec
    );
    println!("  Convergence: {:.1}%", result.convergence_rate * 100.0);
}
