//! Training data schema with COMPLETE wall clock metadata.
//!
//! This module defines the exact JSON schema matching the Python implementation,
//! including ALL runtime awareness fields.

use serde::{Deserialize, Serialize};

/// Complete training example with ALL metadata fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub strategy: String,
    pub render: String,
    pub term: String,
    pub step_k: usize,
    pub target_span: (usize, usize),
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_term: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normal_form: Option<String>,
    pub steps_total: usize,
    pub diverged: bool,
    pub trace_id: String,
    pub meta: ExampleMetadata,
}

/// COMPLETE metadata matching Python implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleMetadata {
    // Term structure
    pub size: usize,
    pub depth: usize,
    pub libs: Vec<String>,
    pub seed: u64,
    pub draw_index: usize,
    pub uid: String,

    // Sharing metrics
    pub thunk_evals: usize,
    pub thunk_hits: usize,

    // Schema tracking
    pub schema_version: String,
    pub term_hash: String,

    // ===== WALL CLOCK RUNTIME METRICS (COMPLETE) =====

    /// Time for THIS specific reduction step in milliseconds
    pub step_ms: f64,

    /// Average step time across entire reduction in milliseconds
    pub avg_step_ms: f64,

    /// Total wall clock time for entire reduction in milliseconds
    pub total_time_ms: f64,

    /// Wall clock budget limit in milliseconds (the constraint)
    pub wall_clock_limit_ms: f64,

    /// Wall clock time remaining in milliseconds
    pub time_remaining_ms: f64,

    /// Fraction of wall clock budget consumed (0.0-1.0)
    pub time_consumed_ratio: f64,

    /// Whether this term is pathological (runtime-based detection)
    pub is_pathological: bool,

    /// Term size growth rate (current_size / initial_size)
    pub size_growth_rate: f64,

    /// Initial term size before reduction
    pub initial_size: usize,
}

impl ExampleMetadata {
    /// Create metadata with all fields populated
    pub fn new(
        size: usize,
        depth: usize,
        seed: u64,
        draw_index: usize,
        trace_id: &str,
        term_hash: &str,
        thunk_evals: usize,
        thunk_hits: usize,
        step_ms: f64,
        avg_step_ms: f64,
        total_time_ms: f64,
        wall_clock_limit_ms: f64,
        time_remaining_ms: f64,
        time_consumed_ratio: f64,
        is_pathological: bool,
        size_growth_rate: f64,
        initial_size: usize,
    ) -> Self {
        ExampleMetadata {
            size,
            depth,
            libs: Vec::new(),
            seed,
            draw_index,
            uid: trace_id.to_string(),
            thunk_evals,
            thunk_hits,
            schema_version: "2.0".to_string(),
            term_hash: term_hash.to_string(),
            step_ms,
            avg_step_ms,
            total_time_ms,
            wall_clock_limit_ms,
            time_remaining_ms,
            time_consumed_ratio,
            is_pathological,
            size_growth_rate,
            initial_size,
        }
    }

    /// Detect if term is pathological based on runtime metrics
    pub fn detect_pathological(
        time_consumed_ratio: f64,
        avg_step_ms: f64,
        size_growth_rate: f64,
        current_size: usize,
    ) -> bool {
        time_consumed_ratio > 0.8    // Used >80% of wall clock budget
            || avg_step_ms > 5.0      // Slow steps (>5ms avg)
            || size_growth_rate > 3.0 // Size tripled
            || current_size > 200     // Very large term
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pathological_detection() {
        // Should detect high time consumption
        assert!(ExampleMetadata::detect_pathological(0.85, 1.0, 1.0, 50));

        // Should detect slow steps
        assert!(ExampleMetadata::detect_pathological(0.5, 6.0, 1.0, 50));

        // Should detect size explosion
        assert!(ExampleMetadata::detect_pathological(0.5, 1.0, 4.0, 50));

        // Should detect large terms
        assert!(ExampleMetadata::detect_pathological(0.5, 1.0, 1.0, 250));

        // Should NOT detect normal cases
        assert!(!ExampleMetadata::detect_pathological(0.3, 1.0, 1.2, 30));
    }

    #[test]
    fn test_serialization() {
        let meta = ExampleMetadata::new(
            10, 3, 42, 0, "test-id", "abc123",
            5, 3,
            0.5, 0.45, 2.5, 100.0, 97.5, 0.025,
            false, 1.2, 10
        );

        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("step_ms"));
        assert!(json.contains("time_consumed_ratio"));
        assert!(json.contains("is_pathological"));
    }
}
