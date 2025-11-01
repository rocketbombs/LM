//! Lambda Calculus generator CLI

use lambda_gen_rs::{GeneratorConfig, ParallelPipeline, PipelineConfig};
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
            let _wall_clock_ms = if args.len() > 5 {
                args[5].parse().expect("Invalid wall_clock_ms")
            } else {
                250.0
            };  // Kept for backward compatibility in CLI args, but not used in classical reduction

            // Use entropy-rich seed for maximum diversity (or user-provided)
            let seed = if args.len() > 6 {
                args[6].parse().expect("Invalid seed")
            } else {
                // Multi-source entropy mixing for TRUE randomness
                use std::time::{SystemTime, UNIX_EPOCH};
                use std::hash::{Hash, Hasher};
                use std::collections::hash_map::DefaultHasher;

                // Source 1: High-resolution timestamp (nanoseconds)
                let time_nanos = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;

                // Source 2: Process ID for uniqueness across runs
                let pid = std::process::id() as u64;

                // Source 3: Hash of command-line args for variation
                let mut hasher = DefaultHasher::new();
                for arg in &args {
                    arg.hash(&mut hasher);
                }
                let args_hash = hasher.finish();

                // Source 4: Memory address entropy (stack location varies)
                let stack_addr = &time_nanos as *const _ as u64;

                // Source 5: Thread ID entropy
                let thread_id = {
                    let mut hasher = DefaultHasher::new();
                    std::thread::current().id().hash(&mut hasher);
                    hasher.finish()
                };

                // Mix all entropy sources with strong avalanche
                let entropy1 = time_nanos
                    .wrapping_mul(0x9e3779b97f4a7c15)  // Golden ratio
                    ^ pid.wrapping_mul(0x6a09e667f3bcc909);  // sqrt(2)

                let entropy2 = args_hash
                    .wrapping_mul(0xbf58476d1ce4e5b9)  // Large prime
                    ^ stack_addr.rotate_left(32);

                let entropy3 = thread_id
                    .wrapping_mul(0x94d049bb133111eb)  // PCG constant
                    ^ (time_nanos >> 32);

                // Final mixing with SplitMix64-style avalanche
                let mut mixed = entropy1 ^ entropy2 ^ entropy3;
                mixed ^= mixed >> 30;
                mixed = mixed.wrapping_mul(0xbf58476d1ce4e5b9);
                mixed ^= mixed >> 27;
                mixed = mixed.wrapping_mul(0x94d049bb133111eb);
                mixed ^= mixed >> 31;

                mixed
            };

            println!("Generating {} examples with {} workers...", num_terms, num_workers);
            println!("Max reduction steps: 1000 per term");
            println!("RNG seed: {} (multi-source entropy for TRUE diversity)", seed);

            let config = PipelineConfig {
                num_workers,
                generator_config: GeneratorConfig {
                    max_depth: 15,        // INCREASED: Push deep nesting harder
                    min_depth: 3,         // Avoid trivial terms
                    max_size: 250,        // INCREASED: Much larger to get 160-200 range
                    allow_divergent: false,  // CRITICAL: Filter out non-normalizing terms
                },
                max_steps: 1000,         // INCREASED: Larger terms need more steps
                strategy: "normal".to_string(),  // Normal-order reduction (leftmost-outermost)
                render: "debruijn".to_string(),
                seed,  // FIX: Use computed seed instead of hardcoded 42
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

            let progress_start = Instant::now();
            pipeline.generate(move |example| {
                let json = example.to_json();
                {
                    let mut w = writer_clone.lock().unwrap();
                    writeln!(w, "{}", json).expect("Failed to write");
                }
                let c = count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if c % 100 == 0 {
                    let elapsed = progress_start.elapsed().as_secs_f64();
                    let throughput = c as f64 / elapsed;
                    print!("\rGenerated {} examples ({:.0} ex/s)...", c, throughput);
                    std::io::stdout().flush().unwrap();
                }
            });

            // Explicitly flush writer before dropping
            {
                let mut w = writer.lock().unwrap();
                w.flush().expect("Failed to flush writer");
            }

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
                max_steps: 1000,
                strategy: "normal".to_string(),
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
