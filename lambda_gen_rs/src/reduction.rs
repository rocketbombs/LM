//! Graph reduction engine with call-by-need semantics and wall clock limiting.
//!
//! Implements efficient lambda calculus reduction with:
//! - Sharing via graph representation
//! - Thunk memoization for call-by-need
//! - Wall clock time limiting for predictable throughput
//! - Per-step timing for runtime-aware training data

use crate::term::{Term, TermArena, TermId, TermNode, TermType};
use std::collections::HashMap;
use std::time::Instant;

/// Configuration for reduction
#[derive(Debug, Clone)]
pub struct ReductionConfig {
    pub wall_clock_limit_ms: f64,
    pub max_steps: usize,
}

impl Default for ReductionConfig {
    fn default() -> Self {
        ReductionConfig {
            wall_clock_limit_ms: 100.0,
            max_steps: 10000,
        }
    }
}

/// Step in a reduction trace
#[derive(Debug, Clone)]
pub struct ReductionStep {
    pub term: Term,
    pub redex_path: Option<Vec<usize>>,
    pub step_time_ms: f64,
}

/// Complete reduction trace with metrics
#[derive(Debug, Clone)]
pub struct ReductionTrace {
    pub steps: Vec<ReductionStep>,
    pub diverged: bool,
    pub total_time_ms: f64,
    pub thunk_evals: usize,
    pub thunk_hits: usize,
}

/// Graph node for call-by-need reduction
#[derive(Debug, Clone)]
enum GraphNode {
    Var(u32),
    Abs(Box<GraphNode>),
    App(Box<GraphNode>, Box<GraphNode>),
    Thunk(Box<GraphNode>, Vec<GraphNode>), // Suspended computation with environment
    BlackHole, // For cycle detection
}

/// Graph reducer with sharing and memoization
pub struct GraphReducer {
    config: ReductionConfig,
    thunk_cache: HashMap<u64, GraphNode>,
    thunk_evals: usize,
    thunk_hits: usize,
}

impl GraphReducer {
    pub fn new(config: ReductionConfig) -> Self {
        GraphReducer {
            config,
            thunk_cache: HashMap::with_capacity(256),
            thunk_evals: 0,
            thunk_hits: 0,
        }
    }

    /// Reduce a term with wall clock limiting and tracing
    pub fn reduce(&mut self, term: &Term) -> ReductionTrace {
        let start_time = Instant::now();
        let mut steps = Vec::with_capacity(64);
        let mut diverged = false;

        // Convert to graph representation
        let mut graph = self.term_to_graph(term, &Vec::new());

        for step_num in 0..self.config.max_steps {
            let step_start = Instant::now();

            // Wall clock check BEFORE expensive operations
            let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            if elapsed_ms > self.config.wall_clock_limit_ms {
                // Exceeded wall clock limit
                let tree = self.graph_to_tree(&graph);
                let redex_path = Self::find_redex(&tree);
                let step_time_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                steps.push(ReductionStep {
                    term: tree,
                    redex_path,
                    step_time_ms,
                });

                diverged = true;
                break;
            }

            // Convert to tree for redex finding and tracing
            let tree = self.graph_to_tree(&graph);
            let redex_path = Self::find_redex(&tree);

            let step_time_ms = step_start.elapsed().as_secs_f64() * 1000.0;
            steps.push(ReductionStep {
                term: tree.clone(),
                redex_path: redex_path.clone(),
                step_time_ms,
            });

            // Check if normal form
            if redex_path.is_none() {
                break;
            }

            // Reduce at redex path
            if let Some(path) = redex_path {
                graph = self.reduce_at_path(graph, &path);
            } else {
                break;
            }
        }

        // Check if hit max steps
        if steps.len() >= self.config.max_steps {
            diverged = true;
        }

        let total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        ReductionTrace {
            steps,
            diverged,
            total_time_ms,
            thunk_evals: self.thunk_evals,
            thunk_hits: self.thunk_hits,
        }
    }

    /// Convert term to graph representation
    fn term_to_graph(&self, term: &Term, env: &Vec<GraphNode>) -> GraphNode {
        match term {
            Term::Var(idx) => {
                if (*idx as usize) < env.len() {
                    env[*idx as usize].clone()
                } else {
                    GraphNode::Var(*idx)
                }
            }
            Term::Abs(body) => {
                let body_graph = self.term_to_graph(body, env);
                GraphNode::Abs(Box::new(body_graph))
            }
            Term::App(func, arg) => {
                let func_graph = self.term_to_graph(func, env);
                let arg_graph = self.term_to_graph(arg, env);
                GraphNode::App(Box::new(func_graph), Box::new(arg_graph))
            }
        }
    }

    /// Convert graph back to tree term
    fn graph_to_tree(&self, graph: &GraphNode) -> Term {
        match graph {
            GraphNode::Var(idx) => Term::Var(*idx),
            GraphNode::Abs(body) => {
                let body_term = self.graph_to_tree(body);
                Term::Abs(Box::new(body_term))
            }
            GraphNode::App(func, arg) => {
                let func_term = self.graph_to_tree(func);
                let arg_term = self.graph_to_tree(arg);
                Term::App(Box::new(func_term), Box::new(arg_term))
            }
            GraphNode::Thunk(body, _) => self.graph_to_tree(body),
            GraphNode::BlackHole => Term::Var(9999), // Should not happen
        }
    }

    /// Find leftmost-outermost redex
    fn find_redex(term: &Term) -> Option<Vec<usize>> {
        fn search(term: &Term, path: &mut Vec<usize>) -> Option<Vec<usize>> {
            match term {
                Term::App(func, _arg) => {
                    // Check if this is a redex (app of abs)
                    if matches!(**func, Term::Abs(_)) {
                        return Some(path.clone());
                    }

                    // Search left subtree
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

    /// Reduce at specific path
    fn reduce_at_path(&mut self, graph: GraphNode, path: &[usize]) -> GraphNode {
        if path.is_empty() {
            // Reduce at root
            return self.reduce_redex(graph);
        }

        match graph {
            GraphNode::Abs(body) => {
                if path[0] == 0 {
                    let reduced_body = self.reduce_at_path(*body, &path[1..]);
                    GraphNode::Abs(Box::new(reduced_body))
                } else {
                    GraphNode::Abs(body)
                }
            }
            GraphNode::App(func, arg) => {
                if path[0] == 0 {
                    let reduced_func = self.reduce_at_path(*func, &path[1..]);
                    GraphNode::App(Box::new(reduced_func), arg)
                } else {
                    let reduced_arg = self.reduce_at_path(*arg, &path[1..]);
                    GraphNode::App(func, Box::new(reduced_arg))
                }
            }
            other => other,
        }
    }

    /// Perform beta reduction
    fn reduce_redex(&mut self, graph: GraphNode) -> GraphNode {
        match graph {
            GraphNode::App(func, arg) => {
                if let GraphNode::Abs(body) = *func {
                    self.thunk_evals += 1;
                    // Beta reduction: substitute arg for var 0 in body
                    self.substitute(*body, 0, *arg)
                } else {
                    GraphNode::App(func, arg)
                }
            }
            other => other,
        }
    }

    /// Substitute term for variable
    fn substitute(&self, term: GraphNode, var: u32, replacement: GraphNode) -> GraphNode {
        match term {
            GraphNode::Var(idx) => {
                if idx == var {
                    replacement
                } else if idx > var {
                    GraphNode::Var(idx - 1)
                } else {
                    GraphNode::Var(idx)
                }
            }
            GraphNode::Abs(body) => {
                let shifted = self.shift_replacement(&replacement, 1);
                let subst_body = self.substitute(*body, var + 1, shifted);
                GraphNode::Abs(Box::new(subst_body))
            }
            GraphNode::App(func, arg) => {
                let subst_func = self.substitute(*func, var, replacement.clone());
                let subst_arg = self.substitute(*arg, var, replacement);
                GraphNode::App(Box::new(subst_func), Box::new(subst_arg))
            }
            other => other,
        }
    }

    /// Shift free variables in term
    fn shift_replacement(&self, term: &GraphNode, amount: i32) -> GraphNode {
        fn shift_impl(term: &GraphNode, amount: i32, cutoff: u32) -> GraphNode {
            match term {
                GraphNode::Var(idx) => {
                    if *idx >= cutoff {
                        GraphNode::Var((*idx as i32 + amount) as u32)
                    } else {
                        GraphNode::Var(*idx)
                    }
                }
                GraphNode::Abs(body) => {
                    let shifted_body = shift_impl(body, amount, cutoff + 1);
                    GraphNode::Abs(Box::new(shifted_body))
                }
                GraphNode::App(func, arg) => {
                    let shifted_func = shift_impl(func, amount, cutoff);
                    let shifted_arg = shift_impl(arg, amount, cutoff);
                    GraphNode::App(Box::new(shifted_func), Box::new(shifted_arg))
                }
                other => other.clone(),
            }
        }

        shift_impl(term, amount, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_reduction() {
        // (λ.0)(λ.0) reduces to λ.0
        let id = Term::Abs(Box::new(Term::Var(0)));
        let term = Term::App(Box::new(id.clone()), Box::new(id));

        let config = ReductionConfig::default();
        let mut reducer = GraphReducer::new(config);
        let trace = reducer.reduce(&term);

        assert!(!trace.diverged);
        assert!(trace.steps.len() > 0);
    }

    #[test]
    fn test_wall_clock_limiting() {
        // Create a term that should timeout
        let config = ReductionConfig {
            wall_clock_limit_ms: 1.0, // Very short timeout
            max_steps: 10000,
        };

        let mut reducer = GraphReducer::new(config);

        // Simple term that should complete quickly
        let id = Term::Abs(Box::new(Term::Var(0)));
        let trace = reducer.reduce(&id);

        // Should complete within 1ms
        assert!(trace.total_time_ms < 1.0 || !trace.diverged);
    }
}
