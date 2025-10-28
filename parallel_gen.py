#!/usr/bin/env python3
"""
Parallel Lambda Calculus term generation using multiprocessing.

Spawns N worker processes that generate terms in parallel, achieving
maximum throughput on multi-core processors.
"""

import sys
import random
import time
import json
from multiprocessing import Process, Queue, Event
from typing import Dict, Any, Optional
from lambda_gen import Config, generate_example


def worker_process(worker_id: int, config: Config, output_queue: Queue,
                   stop_event: Event, seed_offset: int):
    """Worker process that generates terms and sends them to output queue."""
    # Each worker gets a unique seed
    worker_seed = (config.seed or 42) + seed_offset + worker_id
    rng = random.Random(worker_seed)

    draw_index = 0
    consecutive_failures = 0
    max_failures = 100

    while not stop_event.is_set():
        try:
            result = generate_example(config, rng, draw_index)

            if result:
                if isinstance(result, list):
                    for example in result:
                        output_queue.put(('example', example))
                else:
                    output_queue.put(('example', result))

                draw_index += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    output_queue.put(('error', f'Worker {worker_id}: {max_failures} consecutive failures'))
                    break

        except KeyboardInterrupt:
            break
        except Exception as e:
            output_queue.put(('error', f'Worker {worker_id} error: {e}'))
            time.sleep(0.1)  # Back off on errors

    output_queue.put(('worker_done', worker_id))


def parallel_generate(config: Config, num_workers: int, output_file: str,
                     max_terms: Optional[int] = None):
    """
    Generate terms using parallel workers.

    Args:
        config: Generation configuration
        num_workers: Number of parallel worker processes
        output_file: Output JSONL file
        max_terms: Maximum terms to generate (None = unlimited)
    """
    print(f"Starting {num_workers} parallel workers...")
    print(f"Wall clock limit: {config.wall_clock_limit_ms}ms per term")
    print(f"Output: {output_file}")
    print(f"Max terms: {max_terms or 'unlimited'}\n")

    # Create output queue and stop event
    output_queue = Queue(maxsize=num_workers * 100)  # Buffer for smooth flow
    stop_event = Event()

    # Start worker processes
    workers = []
    for i in range(num_workers):
        p = Process(target=worker_process,
                   args=(i, config, output_queue, stop_event, i * 1000))
        p.start()
        workers.append(p)

    # Open output file
    out_file = open(output_file, 'w') if output_file != '-' else sys.stdout

    # Track metrics
    total_examples = 0
    total_errors = 0
    workers_done = 0
    start_time = time.time()
    last_report_time = start_time

    try:
        while workers_done < num_workers:
            # Get items from queue (with timeout to check for Ctrl+C)
            try:
                msg_type, msg_data = output_queue.get(timeout=0.5)
            except:
                # Timeout - check if we should stop
                if max_terms and total_examples >= max_terms:
                    break
                continue

            if msg_type == 'example':
                # Write example to output
                json.dump(msg_data, out_file)
                out_file.write('\n')
                out_file.flush()

                total_examples += 1

                # Progress report every 2 seconds
                now = time.time()
                if now - last_report_time >= 2.0:
                    elapsed = now - start_time
                    rate = total_examples / elapsed if elapsed > 0 else 0
                    sys.stderr.write(f"\r[{total_examples} examples | {rate:.1f}/s | {num_workers} workers]")
                    sys.stderr.flush()
                    last_report_time = now

                # Check if we've hit max terms
                if max_terms and total_examples >= max_terms:
                    break

            elif msg_type == 'error':
                total_errors += 1
                sys.stderr.write(f"\n{msg_data}\n")
                sys.stderr.flush()

            elif msg_type == 'worker_done':
                workers_done += 1
                sys.stderr.write(f"\nWorker {msg_data} finished\n")
                sys.stderr.flush()

    except KeyboardInterrupt:
        sys.stderr.write("\n\nInterrupted! Stopping workers...\n")
        sys.stderr.flush()

    finally:
        # Signal workers to stop
        stop_event.set()

        # Wait for workers to finish (with timeout)
        for p in workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()

        # Drain remaining items from queue
        while not output_queue.empty():
            try:
                msg_type, msg_data = output_queue.get_nowait()
                if msg_type == 'example':
                    json.dump(msg_data, out_file)
                    out_file.write('\n')
                    total_examples += 1
            except:
                break

        # Close output file
        if out_file != sys.stdout:
            out_file.close()

        # Final report
        elapsed = time.time() - start_time
        rate = total_examples / elapsed if elapsed > 0 else 0
        sys.stderr.write(f"\n\n{'='*60}\n")
        sys.stderr.write(f"Parallel Generation Complete\n")
        sys.stderr.write(f"{'='*60}\n")
        sys.stderr.write(f"Total examples: {total_examples}\n")
        sys.stderr.write(f"Total time: {elapsed:.1f}s\n")
        sys.stderr.write(f"Throughput: {rate:.1f} examples/s\n")
        sys.stderr.write(f"Workers: {num_workers}\n")
        sys.stderr.write(f"Per-worker rate: {rate/num_workers:.1f} examples/s\n")
        if total_errors > 0:
            sys.stderr.write(f"Errors: {total_errors}\n")
        sys.stderr.write(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Parallel lambda calculus term generation')
    parser.add_argument('--workers', type=int, default=16,
                       help='Number of parallel workers')
    parser.add_argument('--max-terms', type=int,
                       help='Maximum terms to generate')
    parser.add_argument('--out', default='train.jsonl',
                       help='Output file')

    # Generation parameters
    parser.add_argument('--strategy', default='levy_like', choices=['normal', 'levy_like'])
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--min-depth', type=int, default=2)
    parser.add_argument('--max-size', type=int, default=20)
    parser.add_argument('--wall-clock-limit-ms', type=float, default=100.0)
    parser.add_argument('--share', action='store_true', default=True)
    parser.add_argument('--allow-divergent', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Create config
    config = Config(
        strategy=args.strategy,
        render='debruijn',
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        max_size=args.max_size,
        wall_clock_limit_ms=args.wall_clock_limit_ms,
        share=args.share,
        allow_divergent=args.allow_divergent,
        emit_trace=True,
        seed=args.seed
    )

    # Run parallel generation
    parallel_generate(config, args.workers, args.out, args.max_terms)
