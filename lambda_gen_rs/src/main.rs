//! Lambda Calculus Training Data Generator (Rust)
//!
//! High-performance parallel generation pipeline with complete wall clock metadata.

use clap::{Parser, Subcommand};
use lambda_gen_rs::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "lambda-gen")]
#[command(about = "High-performance Lambda Calculus training data generator", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate training data in parallel
    Generate {
        /// Number of parallel workers (default: number of CPUs)
        #[arg(short, long)]
        workers: Option<usize>,

        /// Maximum number of terms to generate
        #[arg(short, long)]
        max_terms: Option<usize>,

        /// Output file (JSONL format)
        #[arg(short, long, default_value = "train.jsonl")]
        output: String,

        /// Maximum term depth
        #[arg(long, default_value_t = 4)]
        max_depth: usize,

        /// Minimum term depth
        #[arg(long, default_value_t = 2)]
        min_depth: usize,

        /// Maximum term size
        #[arg(long, default_value_t = 20)]
        max_size: usize,

        /// Wall clock limit in milliseconds
        #[arg(long, default_value_t = 50.0)]
        wall_clock_limit_ms: f64,

        /// Allow divergent terms
        #[arg(long)]
        allow_divergent: bool,

        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Run benchmarks
    Bench {
        /// Number of workers to test
        #[arg(short, long, default_value_t = 16)]
        workers: usize,

        /// Number of terms per test
        #[arg(short, long, default_value_t = 1000)]
        terms: usize,
    },
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            workers,
            max_terms,
            output,
            max_depth,
            min_depth,
            max_size,
            wall_clock_limit_ms,
            allow_divergent,
            seed,
        } => {
            run_generation(
                workers,
                max_terms,
                &output,
                max_depth,
                min_depth,
                max_size,
                wall_clock_limit_ms,
                allow_divergent,
                seed,
            )
        }
        Commands::Bench { workers, terms } => run_benchmark(workers, terms),
    }
}

fn run_generation(
    workers: Option<usize>,
    max_terms: Option<usize>,
    output: &str,
    max_depth: usize,
    min_depth: usize,
    max_size: usize,
    wall_clock_limit_ms: f64,
    allow_divergent: bool,
    seed: u64,
) -> std::io::Result<()> {
    let num_workers = workers.unwrap_or_else(num_cpus::get);

    eprintln!("┌─────────────────────────────────────────────────────────┐");
    eprintln!("│   Lambda Calculus Generator (Rust)                     │");
    eprintln!("└─────────────────────────────────────────────────────────┘");
    eprintln!();
    eprintln!("Configuration:");
    eprintln!("  Workers: {}", num_workers);
    eprintln!("  Max terms: {}", max_terms.map_or("unlimited".to_string(), |n| n.to_string()));
    eprintln!("  Depth: {}-{}", min_depth, max_depth);
    eprintln!("  Max size: {}", max_size);
    eprintln!("  Wall clock limit: {}ms", wall_clock_limit_ms);
    eprintln!("  Allow divergent: {}", allow_divergent);
    eprintln!("  Output: {}", output);
    eprintln!();

    // Create output file
    let file = File::create(output)?;
    let mut writer = BufWriter::new(file);

    // Setup pipeline
    let config = PipelineConfig {
        num_workers,
        generator_config: GeneratorConfig {
            max_depth,
            min_depth,
            max_size,
            allow_divergent,
        },
        reduction_config: ReductionConfig {
            wall_clock_limit_ms,
            max_steps: 10000,
        },
        strategy: "levy_like".to_string(),
        render: "debruijn".to_string(),
        seed,
        max_terms,
    };

    let pipeline = ParallelPipeline::new(config);

    // Metrics
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();
    let start_time = Instant::now();
    let last_report = Arc::new(AtomicUsize::new(0));
    let last_report_clone = last_report.clone();

    // Progress reporting thread
    let progress_handle = std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(2));
            let current = count_clone.load(Ordering::Relaxed);
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = current as f64 / elapsed;

            let last = last_report_clone.swap(current, Ordering::Relaxed);
            let recent_rate = (current - last) as f64 / 2.0;

            eprint!("\r[{} examples | {:.1}/s | recent {:.1}/s]",
                current, rate, recent_rate);
            let _ = std::io::stderr().flush();

            if let Some(max) = max_terms {
                if current >= max {
                    break;
                }
            }
        }
    });

    // Generate with callback
    let total = pipeline.generate(|example| {
        // Write as JSONL
        if let Ok(json) = serde_json::to_string(&example) {
            let _ = writeln!(writer, "{}", json);
            count.fetch_add(1, Ordering::Relaxed);
        }
    });

    // Wait for progress thread
    drop(progress_handle);

    writer.flush()?;

    let elapsed = start_time.elapsed().as_secs_f64();
    let throughput = total as f64 / elapsed;

    eprintln!("\n");
    eprintln!("┌─────────────────────────────────────────────────────────┐");
    eprintln!("│   Generation Complete                                   │");
    eprintln!("└─────────────────────────────────────────────────────────┘");
    eprintln!();
    eprintln!("Results:");
    eprintln!("  Total examples: {}", total);
    eprintln!("  Time: {:.2}s", elapsed);
    eprintln!("  Throughput: {:.1} examples/s", throughput);
    eprintln!("  Per-worker: {:.1} examples/s", throughput / num_workers as f64);
    eprintln!();

    Ok(())
}

fn run_benchmark(workers: usize, terms: usize) -> std::io::Result<()> {
    eprintln!("┌─────────────────────────────────────────────────────────┐");
    eprintln!("│   Throughput Benchmark                                  │");
    eprintln!("└─────────────────────────────────────────────────────────┘");
    eprintln!();

    let worker_counts = vec![1, 2, 4, 8, workers];

    for num_workers in worker_counts {
        eprintln!("Testing {} workers...", num_workers);

        let config = PipelineConfig {
            num_workers,
            max_terms: Some(terms),
            ..Default::default()
        };

        let pipeline = ParallelPipeline::new(config);
        let start = Instant::now();

        let total = pipeline.generate(|_| {
            // Discard output for benchmark
        });

        let elapsed = start.elapsed().as_secs_f64();
        let throughput = total as f64 / elapsed;

        eprintln!("  {} workers: {:.1} examples/s ({:.1} per-worker)",
            num_workers, throughput, throughput / num_workers as f64);
    }

    eprintln!();

    Ok(())
}
