//! Term rendering to De Bruijn notation with span tracking.
//!
//! Converts terms to string representation while tracking character positions
//! for each subterm. Essential for span-based training data.

use crate::term::Term;
use std::collections::HashMap;

/// Result of rendering with span information
#[derive(Debug, Clone)]
pub struct RenderResult {
    pub string: String,
    pub spans: HashMap<usize, (usize, usize)>, // node_id -> (start_char, end_char)
}

/// Render term to De Bruijn notation with span tracking
pub fn render_debruijn(term: &Term) -> RenderResult {
    let mut string = String::with_capacity(256);
    let mut spans = HashMap::new();
    let mut node_id = 1;

    render_term_impl(term, &mut string, &mut spans, &mut node_id);

    RenderResult { string, spans }
}

fn render_term_impl(
    term: &Term,
    output: &mut String,
    spans: &mut HashMap<usize, (usize, usize)>,
    node_id: &mut usize,
) {
    let start_pos = output.len();
    let current_node = *node_id;
    *node_id += 1;

    match term {
        Term::Var(idx) => {
            output.push_str(&idx.to_string());
        }
        Term::Abs(body) => {
            output.push('\\');
            output.push('.');
            render_term_impl(body, output, spans, node_id);
        }
        Term::App(func, arg) => {
            // Determine if we need parentheses
            let need_func_parens = matches!(**func, Term::Abs(_));
            let need_arg_parens = !matches!(**arg, Term::Var(_));

            if need_func_parens {
                output.push('(');
            }
            render_term_impl(func, output, spans, node_id);
            if need_func_parens {
                output.push(')');
            }

            if need_arg_parens {
                output.push('(');
            }
            render_term_impl(arg, output, spans, node_id);
            if need_arg_parens {
                output.push(')');
            }
        }
    }

    let end_pos = output.len();
    spans.insert(current_node, (start_pos, end_pos));
}

/// Find redex span from path and render result
pub fn get_redex_span(
    term: &Term,
    redex_path: Option<&[usize]>,
    render_result: &RenderResult,
) -> (usize, usize) {
    match redex_path {
        None => (0, 0), // Normal form
        Some(path) => {
            let node_id = path_to_node_id(path);
            render_result.spans.get(&node_id).copied().unwrap_or((0, 0))
        }
    }
}

/// Convert path to node ID in rendering tree
fn path_to_node_id(path: &[usize]) -> usize {
    let mut node_id = 1;
    for &direction in path {
        if direction == 0 {
            node_id = node_id * 2 + 1;
        } else {
            node_id = node_id * 2 + 2;
        }
    }
    node_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_var() {
        let term = Term::Var(0);
        let result = render_debruijn(&term);
        assert_eq!(result.string, "0");
    }

    #[test]
    fn test_render_abs() {
        let term = Term::Abs(Box::new(Term::Var(0)));
        let result = render_debruijn(&term);
        assert_eq!(result.string, "\\.0");
    }

    #[test]
    fn test_render_app() {
        // (λ.0)(λ.0)
        let id = Term::Abs(Box::new(Term::Var(0)));
        let term = Term::App(Box::new(id.clone()), Box::new(id));
        let result = render_debruijn(&term);
        assert_eq!(result.string, "(\\.0)(\\.0)");
    }

    #[test]
    fn test_spans_tracked() {
        let term = Term::Abs(Box::new(Term::Var(0)));
        let result = render_debruijn(&term);

        // Should have spans for both the abs and the var
        assert!(result.spans.len() >= 2);
    }
}
