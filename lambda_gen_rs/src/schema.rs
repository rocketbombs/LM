//! Training data schema with COMPLETE wall clock metadata.
//!
//! This module defines the exact JSON schema matching the Python implementation,
//! including ALL runtime awareness fields.

/// Complete training example with ALL metadata fields
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub strategy: String,
    pub render: String,
    pub term: String,
    pub step_k: usize,
    pub target_span: (usize, usize),
    pub next_term: Option<String>,
    pub normal_form: Option<String>,
    pub steps_total: usize,
    pub diverged: bool,
    pub trace_id: String,
    pub meta: ExampleMetadata,
}

/// COMPLETE metadata matching Python implementation
#[derive(Debug, Clone)]
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
        time_consumed_ratio > 0.8    // Used >80% of wall clock budget (more permissive)
            || avg_step_ms > 5.0      // Slow steps (>5ms avg, more permissive)
            || size_growth_rate > 3.5 // Size more than tripled (more permissive for growth examples)
            || current_size > 250     // INCREASED: Allow terms up to 250 nodes (was 150)
    }
}

impl TrainingExample {
    /// Serialize to JSONL format
    pub fn to_json(&self) -> String {
        let mut json = String::from("{");

        json.push_str(&format!("\"strategy\":\"{}\",", escape_json(&self.strategy)));
        json.push_str(&format!("\"render\":\"{}\",", escape_json(&self.render)));
        json.push_str(&format!("\"term\":\"{}\",", escape_json(&self.term)));
        json.push_str(&format!("\"step_k\":{},", self.step_k));
        json.push_str(&format!("\"target_span\":[{},{}],", self.target_span.0, self.target_span.1));

        if let Some(ref next) = self.next_term {
            json.push_str(&format!("\"next_term\":\"{}\",", escape_json(next)));
        } else {
            json.push_str("\"next_term\":null,");
        }

        if let Some(ref nf) = self.normal_form {
            json.push_str(&format!("\"normal_form\":\"{}\",", escape_json(nf)));
        } else {
            json.push_str("\"normal_form\":null,");
        }

        json.push_str(&format!("\"steps_total\":{},", self.steps_total));
        json.push_str(&format!("\"diverged\":{},", self.diverged));
        json.push_str(&format!("\"trace_id\":\"{}\",", escape_json(&self.trace_id)));

        // Metadata
        json.push_str("\"meta\":{");
        json.push_str(&format!("\"size\":{},", self.meta.size));
        json.push_str(&format!("\"depth\":{},", self.meta.depth));
        json.push_str("\"libs\":[");
        for (i, lib) in self.meta.libs.iter().enumerate() {
            if i > 0 { json.push(','); }
            json.push_str(&format!("\"{}\"", escape_json(lib)));
        }
        json.push_str("],");
        json.push_str(&format!("\"seed\":{},", self.meta.seed));
        json.push_str(&format!("\"draw_index\":{},", self.meta.draw_index));
        json.push_str(&format!("\"uid\":\"{}\",", escape_json(&self.meta.uid)));
        json.push_str(&format!("\"thunk_evals\":{},", self.meta.thunk_evals));
        json.push_str(&format!("\"thunk_hits\":{},", self.meta.thunk_hits));
        json.push_str(&format!("\"schema_version\":\"{}\",", escape_json(&self.meta.schema_version)));
        json.push_str(&format!("\"term_hash\":\"{}\",", escape_json(&self.meta.term_hash)));
        json.push_str(&format!("\"step_ms\":{},", self.meta.step_ms));
        json.push_str(&format!("\"avg_step_ms\":{},", self.meta.avg_step_ms));
        json.push_str(&format!("\"total_time_ms\":{},", self.meta.total_time_ms));
        json.push_str(&format!("\"wall_clock_limit_ms\":{},", self.meta.wall_clock_limit_ms));
        json.push_str(&format!("\"time_remaining_ms\":{},", self.meta.time_remaining_ms));
        json.push_str(&format!("\"time_consumed_ratio\":{},", self.meta.time_consumed_ratio));
        json.push_str(&format!("\"is_pathological\":{},", self.meta.is_pathological));
        json.push_str(&format!("\"size_growth_rate\":{},", self.meta.size_growth_rate));
        json.push_str(&format!("\"initial_size\":{}", self.meta.initial_size));
        json.push_str("}");

        json.push('}');
        json
    }
}

/// Escape JSON strings
fn escape_json(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            '\n' => vec!['\\', 'n'],
            '\r' => vec!['\\', 'r'],
            '\t' => vec!['\\', 't'],
            c => vec![c],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pathological_detection() {
        // Should detect high time consumption (>80%)
        assert!(ExampleMetadata::detect_pathological(0.85, 1.0, 1.0, 50));

        // Should detect slow steps (>5ms avg)
        assert!(ExampleMetadata::detect_pathological(0.5, 6.0, 1.0, 50));

        // Should detect size explosion (>3.5x growth)
        assert!(ExampleMetadata::detect_pathological(0.5, 1.0, 4.0, 50));

        // Should detect very large terms (>250 nodes)
        assert!(ExampleMetadata::detect_pathological(0.5, 1.0, 1.0, 260));

        // Should NOT detect normal cases
        assert!(!ExampleMetadata::detect_pathological(0.3, 1.0, 1.2, 30));

        // Should NOT detect large but acceptable terms (150-250 nodes)
        assert!(!ExampleMetadata::detect_pathological(0.5, 2.0, 1.5, 200));
    }

    #[test]
    fn test_serialization() {
        let meta = ExampleMetadata::new(
            10, 3, 42, 0, "test-id", "abc123",
            5, 3,
            0.5, 0.45, 2.5, 100.0, 97.5, 0.025,
            false, 1.2, 10
        );

        let example = TrainingExample {
            strategy: "levy_like".to_string(),
            render: "debruijn".to_string(),
            term: "\\.(\\.(10))".to_string(),
            step_k: 0,
            target_span: (0, 10),
            next_term: None,
            normal_form: None,
            steps_total: 5,
            diverged: false,
            trace_id: "test-trace".to_string(),
            meta,
        };

        let json = example.to_json();
        assert!(json.contains("step_ms"));
        assert!(json.contains("time_consumed_ratio"));
        assert!(json.contains("is_pathological"));
    }
}
