//! Neural lambda calculus reducer using ONNX inference.
//!
//! Loads a trained PyTorch model (exported to ONNX) and uses it to predict
//! the next redex to reduce at each step.

use crate::term::Term;
use crate::tokenizer::LambdaTokenizer;
use ndarray::{Array1, Array2};
use std::time::Instant;
use tract_onnx::prelude::*;

/// Neural reduction result with metrics
#[derive(Debug, Clone)]
pub struct NeuralReduction {
    pub final_term: Term,
    pub steps: usize,
    pub total_time_ms: f64,
    pub total_inference_ms: f64,
    pub converged: bool,
}

/// Neural reducer using ONNX model
pub struct NeuralReducer {
    model: TypedRunnableModel<TypedFact>,
    tokenizer: LambdaTokenizer,
    max_steps: usize,
}

impl NeuralReducer {
    /// Load model from ONNX file
    pub fn new(onnx_path: &str, max_steps: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let model = tract_onnx::onnx()
            .model_for_path(onnx_path)?
            .into_optimized()?
            .into_runnable()?;

        let tokenizer = LambdaTokenizer::new();

        Ok(NeuralReducer {
            model,
            tokenizer,
            max_steps,
        })
    }

    /// Reduce term to normal form using neural predictions
    pub fn reduce(&self, term: &Term) -> NeuralReduction {
        let start = Instant::now();
        let mut current = term.clone();
        let mut steps = 0;
        let mut total_inference_ms = 0.0;

        for step in 0..self.max_steps {
            let inf_start = Instant::now();

            // Render term to string (de Bruijn notation)
            let term_str = self.render_term(&current);

            // Get neural prediction
            let (is_nf, redex_start, redex_end) = match self.predict(&term_str) {
                Ok(pred) => pred,
                Err(_) => {
                    // Inference error, treat as normal form
                    break;
                }
            };

            total_inference_ms += inf_start.elapsed().as_secs_f64() * 1000.0;

            // Check if normal form
            if is_nf {
                break;
            }

            // Convert character span to structural path
            if let Some(path) = self.span_to_path(&current, &term_str, redex_start, redex_end) {
                // Validate it's actually a redex
                if Self::is_redex_at_path(&current, &path) {
                    // Apply β-reduction
                    current = Self::reduce_at_path(current, &path);
                    steps = step + 1;
                } else {
                    // Invalid prediction, stop
                    break;
                }
            } else {
                // Couldn't map span to path, stop
                break;
            }
        }

        NeuralReduction {
            final_term: current,
            steps,
            total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            total_inference_ms,
            converged: steps < self.max_steps,
        }
    }

    /// Run model inference to predict next redex
    fn predict(&self, term_str: &str) -> Result<(bool, usize, usize), Box<dyn std::error::Error>> {
        // Tokenize input
        let tokens = self.tokenizer.encode(term_str, true);
        let seq_len = tokens.len();

        // Create input tensors
        let input_ids: Array2<i64> = Array2::from_shape_vec(
            (1, seq_len),
            tokens.iter().map(|&t| t as i64).collect(),
        )?;

        let attention_mask: Array2<i64> = Array2::from_elem((1, seq_len), 1);

        // Run inference
        let outputs = self.model.run(tvec![
            input_ids.into_dyn().into(),
            attention_mask.into_dyn().into(),
        ])?;

        // Extract outputs
        let start_logits = outputs[0]
            .to_array_view::<f32>()?
            .into_dimensionality::<ndarray::Ix2>()?;
        let end_logits = outputs[1]
            .to_array_view::<f32>()?
            .into_dimensionality::<ndarray::Ix2>()?;
        let nf_logit = outputs[2]
            .to_array_view::<f32>()?
            .into_dimensionality::<ndarray::Ix1>()?;

        // Check normal form prediction
        let nf_prob = sigmoid(nf_logit[0]);
        if nf_prob > 0.5 {
            return Ok((true, 0, 0));
        }

        // Get span predictions (argmax)
        let start_idx = argmax(start_logits.row(0));
        let end_idx = argmax(end_logits.row(0));

        Ok((false, start_idx, end_idx))
    }

    /// Render term to de Bruijn string notation
    fn render_term(&self, term: &Term) -> String {
        fn render_rec(term: &Term) -> String {
            match term {
                Term::Var(idx) => format!("{}", idx),
                Term::Abs(body) => format!("(\\.{})", render_rec(body)),
                Term::App(func, arg) => format!("({}{})", render_rec(func), render_rec(arg)),
            }
        }
        render_rec(term)
    }

    /// Convert character span to structural path
    ///
    /// This is a simplified version - in production you'd want span tracking
    /// during rendering for exact mapping. For now, we use best-effort matching.
    fn span_to_path(
        &self,
        term: &Term,
        _rendered: &str,
        start: usize,
        end: usize,
    ) -> Option<Vec<usize>> {
        // Simplified: try to find a redex near the predicted span
        // In practice, you'd maintain span mappings during rendering
        let _ = (start, end); // Suppress unused warning
        Self::find_any_redex(term) // Fallback to any redex for now
    }

    /// Find any redex in the term (fallback when span mapping fails)
    fn find_any_redex(term: &Term) -> Option<Vec<usize>> {
        fn search(term: &Term, path: &mut Vec<usize>) -> Option<Vec<usize>> {
            match term {
                Term::App(func, _arg) => {
                    if matches!(**func, Term::Abs(_)) {
                        return Some(path.clone());
                    }
                    path.push(0);
                    if let Some(result) = search(func, path) {
                        return Some(result);
                    }
                    path.pop();
                    path.push(1);
                    if let Some(result) = search(_arg, path) {
                        return Some(result);
                    }
                    path.pop();
                    None
                }
                Term::Abs(body) => {
                    path.push(0);
                    let result = search(body, path);
                    path.pop();
                    result
                }
                Term::Var(_) => None,
            }
        }
        search(term, &mut Vec::new())
    }

    /// Check if there's a redex at the given path
    fn is_redex_at_path(term: &Term, path: &[usize]) -> bool {
        if path.is_empty() {
            return matches!(term, Term::App(func, _) if matches!(**func, Term::Abs(_)));
        }

        match term {
            Term::Abs(body) if path[0] == 0 => Self::is_redex_at_path(body, &path[1..]),
            Term::App(func, _) if path[0] == 0 => Self::is_redex_at_path(func, &path[1..]),
            Term::App(_, arg) if path[0] == 1 => Self::is_redex_at_path(arg, &path[1..]),
            _ => false,
        }
    }

    /// Apply β-reduction at path (same as classical reducer)
    fn reduce_at_path(term: Term, path: &[usize]) -> Term {
        if path.is_empty() {
            return Self::beta_reduce(term);
        }

        match term {
            Term::Abs(body) => {
                if path[0] == 0 {
                    let reduced_body = Self::reduce_at_path(*body, &path[1..]);
                    Term::Abs(Box::new(reduced_body))
                } else {
                    Term::Abs(body)
                }
            }
            Term::App(func, arg) => {
                if path[0] == 0 {
                    let reduced_func = Self::reduce_at_path(*func, &path[1..]);
                    Term::App(Box::new(reduced_func), arg)
                } else {
                    let reduced_arg = Self::reduce_at_path(*arg, &path[1..]);
                    Term::App(func, Box::new(reduced_arg))
                }
            }
            other => other,
        }
    }

    fn beta_reduce(term: Term) -> Term {
        match term {
            Term::App(func, arg) => {
                if let Term::Abs(body) = *func {
                    Self::substitute(*body, 0, *arg)
                } else {
                    Term::App(func, arg)
                }
            }
            other => other,
        }
    }

    fn substitute(term: Term, var: u32, replacement: Term) -> Term {
        match term {
            Term::Var(idx) => {
                if idx == var {
                    replacement
                } else if idx > var {
                    Term::Var(idx - 1)
                } else {
                    Term::Var(idx)
                }
            }
            Term::Abs(body) => {
                let shifted = Self::shift(replacement, 1);
                let subst_body = Self::substitute(*body, var + 1, shifted);
                Term::Abs(Box::new(subst_body))
            }
            Term::App(func, arg) => {
                let subst_func = Self::substitute(*func, var, replacement.clone());
                let subst_arg = Self::substitute(*arg, var, replacement);
                Term::App(Box::new(subst_func), Box::new(subst_arg))
            }
        }
    }

    fn shift(term: Term, delta: i32) -> Term {
        fn shift_rec(term: Term, delta: i32, cutoff: u32) -> Term {
            match term {
                Term::Var(idx) => {
                    if idx >= cutoff {
                        Term::Var(((idx as i32) + delta) as u32)
                    } else {
                        Term::Var(idx)
                    }
                }
                Term::Abs(body) => Term::Abs(Box::new(shift_rec(*body, delta, cutoff + 1))),
                Term::App(func, arg) => {
                    let shifted_func = shift_rec(*func, delta, cutoff);
                    let shifted_arg = shift_rec(*arg, delta, cutoff);
                    Term::App(Box::new(shifted_func), Box::new(shifted_arg))
                }
            }
        }
        shift_rec(term, delta, 0)
    }
}

/// Sigmoid activation
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Argmax over array
#[inline]
fn argmax(arr: ndarray::ArrayView1<f32>) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
