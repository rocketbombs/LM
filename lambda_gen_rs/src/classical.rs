//! Classical normal-order β-reduction (leftmost-outermost strategy).
//!
//! Pure tree-based reduction with no sharing or memoization.
//! This is the baseline for comparing against neural reduction.

use crate::term::Term;
use std::time::Instant;

/// Single step in a reduction trace
#[derive(Debug, Clone)]
pub struct ReductionStep {
    pub term: Term,
    pub redex_path: Option<Vec<usize>>,
}

/// Reduction result with metrics
#[derive(Debug, Clone)]
pub struct ClassicalReduction {
    pub final_term: Term,
    pub steps: usize,
    pub total_time_ms: f64,
    pub converged: bool,
}

/// Reduction trace with all intermediate steps
#[derive(Debug, Clone)]
pub struct ClassicalTrace {
    pub steps: Vec<ReductionStep>,
    pub converged: bool,
    pub total_time_ms: f64,
}

/// Classical normal-order reducer
pub struct ClassicalReducer {
    max_steps: usize,
}

impl ClassicalReducer {
    pub fn new(max_steps: usize) -> Self {
        ClassicalReducer { max_steps }
    }

    /// Reduce term to normal form
    pub fn reduce(&self, term: &Term) -> ClassicalReduction {
        let start = Instant::now();
        let mut current = term.clone();
        let mut steps = 0;

        for step in 0..self.max_steps {
            // Find leftmost-outermost redex
            if let Some(path) = Self::find_redex(&current) {
                // Apply β-reduction at path
                current = Self::reduce_at_path(current, &path);
                steps = step + 1;
            } else {
                // Normal form reached
                return ClassicalReduction {
                    final_term: current,
                    steps,
                    total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                    converged: true,
                };
            }
        }

        // Hit max steps without converging
        ClassicalReduction {
            final_term: current,
            steps,
            total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            converged: false,
        }
    }

    /// Reduce term with streaming callback for training data generation
    ///
    /// MEMORY EFFICIENT: Processes each step via callback without accumulating trace.
    /// Callback is invoked for each step with: (step_index, term, redex_path, is_final)
    /// Returns (converged, total_steps, total_time_ms)
    pub fn reduce_with_streaming<F>(&self, term: &Term, mut callback: F) -> (bool, usize, f64)
    where
        F: FnMut(usize, &Term, Option<&[usize]>, bool) -> bool, // Returns false to abort
    {
        let start = Instant::now();
        let mut current = term.clone();
        let mut step_count = 0;

        // Process initial state
        let initial_redex = Self::find_redex(&current);
        let is_nf = initial_redex.is_none();

        // Invoke callback for initial step
        if !callback(0, &current, initial_redex.as_deref(), is_nf) {
            // Callback requested abort
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            return (false, 1, elapsed);
        }

        // If already in normal form, return immediately
        if is_nf {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            return (true, 1, elapsed);
        }

        // Reduce step by step
        for step in 0..self.max_steps {
            if let Some(path) = Self::find_redex(&current) {
                // Apply β-reduction at path
                current = Self::reduce_at_path(current, &path);
                step_count = step + 1;

                // Find next redex
                let next_redex = Self::find_redex(&current);
                let is_final = next_redex.is_none();

                // Invoke callback with reduced term
                if !callback(step_count, &current, next_redex.as_deref(), is_final) {
                    // Callback requested abort
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    return (false, step_count + 1, elapsed);
                }

                // Check if we reached normal form
                if is_final {
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    return (true, step_count + 1, elapsed);
                }
            } else {
                // Normal form reached
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                return (true, step_count + 1, elapsed);
            }
        }

        // Hit max steps without converging
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        (false, step_count + 1, elapsed)
    }

    /// Reduce term with full trace for training data generation
    ///
    /// DEPRECATED: Use reduce_with_streaming for memory efficiency.
    /// This method accumulates entire trace in memory.
    #[deprecated(note = "Use reduce_with_streaming to avoid memory leaks")]
    pub fn reduce_with_trace(&self, term: &Term) -> ClassicalTrace {
        let start = Instant::now();
        let mut current = term.clone();
        let mut trace = Vec::with_capacity(64);

        // Add initial state
        let initial_redex = Self::find_redex(&current);
        trace.push(ReductionStep {
            term: current.clone(),
            redex_path: initial_redex.clone(),
        });

        // If already in normal form, return immediately
        if initial_redex.is_none() {
            return ClassicalTrace {
                steps: trace,
                converged: true,
                total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            };
        }

        // Reduce step by step
        for _ in 0..self.max_steps {
            if let Some(path) = Self::find_redex(&current) {
                // Apply β-reduction at path
                current = Self::reduce_at_path(current, &path);

                // Find next redex
                let next_redex = Self::find_redex(&current);
                trace.push(ReductionStep {
                    term: current.clone(),
                    redex_path: next_redex.clone(),
                });

                // Check if we reached normal form
                if next_redex.is_none() {
                    return ClassicalTrace {
                        steps: trace,
                        converged: true,
                        total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                    };
                }
            } else {
                // Normal form reached
                return ClassicalTrace {
                    steps: trace,
                    converged: true,
                    total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                };
            }
        }

        // Hit max steps without converging
        ClassicalTrace {
            steps: trace,
            converged: false,
            total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Find leftmost-outermost redex (normal-order)
    fn find_redex(term: &Term) -> Option<Vec<usize>> {
        fn search(term: &Term, path: &mut Vec<usize>) -> Option<Vec<usize>> {
            match term {
                Term::App(func, _arg) => {
                    // Check if this is a redex (app of abs)
                    if matches!(**func, Term::Abs(_)) {
                        return Some(path.clone());
                    }

                    // Search left subtree (outermost-first)
                    path.push(0);
                    if let Some(result) = search(func, path) {
                        return Some(result);
                    }
                    path.pop();

                    // Search right subtree
                    path.push(1);
                    if let Some(result) = search(_arg, path) {
                        return Some(result);
                    }
                    path.pop();

                    None
                }
                Term::Abs(body) => {
                    // Search inside abstraction
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

    /// Apply β-reduction at specific path
    fn reduce_at_path(term: Term, path: &[usize]) -> Term {
        if path.is_empty() {
            // Reduce at root
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

    /// Perform β-reduction: (λ.body) arg → body[0 := arg]
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

    /// Substitute term for variable: body[var := replacement]
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

    /// Shift free variables by delta
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
                Term::Abs(body) => {
                    Term::Abs(Box::new(shift_rec(*body, delta, cutoff + 1)))
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_reduction() {
        // (λ.0) x → x
        let id = Term::Abs(Box::new(Term::Var(0)));
        let x = Term::Var(0);
        let term = Term::App(Box::new(id), Box::new(x.clone()));

        let reducer = ClassicalReducer::new(100);
        let result = reducer.reduce(&term);

        assert!(result.converged);
        assert_eq!(result.steps, 1);
        assert!(matches!(result.final_term, Term::Var(0)));
    }

    #[test]
    fn test_church_numeral() {
        // 2 = λf.λx.f(f x)
        // 2 f x → f (f x)
        let two = Term::Abs(Box::new(Term::Abs(Box::new(Term::App(
            Box::new(Term::Var(1)),
            Box::new(Term::App(
                Box::new(Term::Var(1)),
                Box::new(Term::Var(0)),
            )),
        )))));

        let f = Term::Var(1);
        let x = Term::Var(0);

        // Apply 2 to f and x
        let term = Term::App(
            Box::new(Term::App(Box::new(two), Box::new(f))),
            Box::new(x),
        );

        let reducer = ClassicalReducer::new(100);
        let result = reducer.reduce(&term);

        assert!(result.converged);
        assert_eq!(result.steps, 2); // Two beta reductions
    }
}
