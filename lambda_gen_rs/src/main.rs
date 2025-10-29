//! Lambda Calculus generator CLI

use lambda_gen_rs::{GeneratorConfig, ParallelPipeline, PipelineConfig, ReductionConfig};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <command> [options]", args[0]);
        eprintln!("Commands:");
        eprintln!("  generate <output_file> <num_terms> [num_workers] [wall_clock_ms]");
        eprintln!("  benchmark <num_terms> [num_workers]");
        std::process::exit(1);
    }

    let command = &args[1];

    match command.as_str() {
        "generate" => {
            if args.len() < 4 {
                eprintln!("Usage: {} generate <output_file> <num_terms> [num_workers] [wall_clock_ms]", args[0]);
                std::process::exit(1);
            }

            let output_file = &args[2];
            let num_terms: usize = args[3].parse().expect("Invalid num_terms");
            let num_workers = if args.len() > 4 {
                args[4].parse().expect("Invalid num_workers")
            } else {
                8
            };
            let wall_clock_ms = if args.len() > 5 {
                args[5].parse().expect("Invalid wall_clock_ms")
            } else {
                100.0
            };

            println!("Generating {} terms with {} workers...", num_terms, num_workers);
            println!("Wall clock limit: {}ms per term", wall_clock_ms);

            let config = PipelineConfig {
                num_workers,
                generator_config: GeneratorConfig {
                    max_depth: 8,         // Balanced: allows complex terms without explosion
                    min_depth: 3,         // Avoid trivial terms
                    max_size: 100,        // Initial size before reduction growth
                    allow_divergent: false,  // CRITICAL: Filter out non-normalizing terms
                },
                reduction_config: ReductionConfig {
                    wall_clock_limit_ms: wall_clock_ms,
                    max_steps: 500,      // Normalizing terms complete in <500 steps
                },
                strategy: "levy_like".to_string(),
                render: "debruijn".to_string(),
                seed: 42,
                max_terms: Some(num_terms),
            };

            let pipeline = ParallelPipeline::new(config);
            let start = Instant::now();

            let file = File::create(output_file).expect("Failed to create output file");
            let writer = BufWriter::new(file);
            let writer = std::sync::Arc::new(std::sync::Mutex::new(writer));
            let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

            let writer_clone = writer.clone();
            let count_clone = count.clone();

            pipeline.generate(move |example| {
                let json = example.to_json();
                {
                    let mut w = writer_clone.lock().unwrap();
                    writeln!(w, "{}", json).expect("Failed to write");
                }
                let c = count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if c % 100 == 0 {
                    print!("\rGenerated {} examples...", c);
                    std::io::stdout().flush().unwrap();
                }
            });

            let final_count = count.load(std::sync::atomic::Ordering::Relaxed);
            let elapsed = start.elapsed();
            println!("\n\nGenerated {} examples in {:.2}s", final_count, elapsed.as_secs_f64());
            println!("Throughput: {:.2} examples/s", final_count as f64 / elapsed.as_secs_f64());
        }

        "benchmark" => {
            if args.len() < 3 {
                eprintln!("Usage: {} benchmark <num_terms> [num_workers]", args[0]);
                std::process::exit(1);
            }

            let num_terms: usize = args[2].parse().expect("Invalid num_terms");
            let num_workers = if args.len() > 3 {
                args[3].parse().expect("Invalid num_workers")
            } else {
                8
            };

            println!("Benchmarking with {} terms and {} workers...", num_terms, num_workers);

            let config = PipelineConfig {
                num_workers,
                generator_config: GeneratorConfig::default(),
                reduction_config: ReductionConfig::default(),
                strategy: "levy_like".to_string(),
                render: "debruijn".to_string(),
                seed: 42,
                max_terms: Some(num_terms),
            };

            let pipeline = ParallelPipeline::new(config);
            let start = Instant::now();

            let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let count_clone = count.clone();

            pipeline.generate(move |_example| {
                let c = count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if c % 100 == 0 {
                    print!("\rProcessed {} examples...", c);
                    std::io::stdout().flush().unwrap();
                }
            });

            let final_count = count.load(std::sync::atomic::Ordering::Relaxed);
            let elapsed = start.elapsed();
            println!("\n\nProcessed {} examples in {:.2}s", final_count, elapsed.as_secs_f64());
            println!("Throughput: {:.2} examples/s", final_count as f64 / elapsed.as_secs_f64());
        }

        _ => {
            eprintln!("Unknown command: {}", command);
            eprintln!("Available commands: generate, benchmark");
            std::process::exit(1);
        }
    }
}
