//! High-performance Lambda Calculus term generation and reduction.
//!
//! This library provides:
//! - Efficient term representation with arena allocation
//! - Graph reduction with call-by-need semantics and sharing
//! - Wall clock-limited reduction for predictable throughput
//! - Lock-free parallel generation pipeline
//! - Runtime-aware training data generation
//!
//! # Performance
//!
//! Optimized for maximum throughput:
//! - Zero-copy term representation
//! - Lock-free parallel workers
//! - Arena allocation to minimize GC pressure
//! - True parallelism without GIL limitations

pub mod term;
pub mod reduction;
pub mod generator;
pub mod parallel;
pub mod render;
pub mod schema;
pub mod classical;
pub mod tokenizer;
pub mod neural;

pub use term::{Term, TermType, TermArena};
pub use reduction::{GraphReducer, ReductionConfig, ReductionTrace};
pub use generator::{TermGenerator, GeneratorConfig};
pub use parallel::{ParallelPipeline, PipelineConfig};
pub use schema::TrainingExample;
pub use classical::ClassicalReducer;
pub use tokenizer::LambdaTokenizer;
pub use neural::NeuralReducer;
