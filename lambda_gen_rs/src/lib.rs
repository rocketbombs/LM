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

// Neural module disabled - requires external dependencies (tract_onnx, ndarray)
// Uncomment when dependencies are available
// pub mod neural;

pub use term::{Term, TermType, TermArena};
pub use generator::{TermGenerator, GeneratorConfig};
pub use parallel::{ParallelPipeline, PipelineConfig};
pub use schema::TrainingExample;
pub use classical::{ClassicalReducer, ClassicalReduction, ClassicalTrace, ReductionStep};
pub use tokenizer::LambdaTokenizer;

// NeuralReducer disabled - requires external dependencies
// pub use neural::NeuralReducer;

// Keep reduction module for backward compatibility if needed, but mark as deprecated
#[allow(deprecated)]
pub use reduction::{GraphReducer, ReductionConfig};
