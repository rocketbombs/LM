//! Lambda calculus term representation with arena allocation.
//!
//! Terms are stored in an arena for efficient allocation and deallocation.
//! Uses indices instead of pointers for cache-friendly access.

use std::fmt;

/// Term type discriminant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermType {
    Var,
    Abs,
    App,
}

/// Compact term representation using indices into arena
#[derive(Debug, Clone, Copy)]
pub struct TermId(pub u32);

impl TermId {
    #[inline]
    pub const fn new(id: u32) -> Self {
        TermId(id)
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Term node stored in arena
#[derive(Debug, Clone)]
pub struct TermNode {
    pub ty: TermType,
    pub var_index: u32,      // For Var
    pub left: Option<TermId>, // For App (function) or Abs (body)
    pub right: Option<TermId>, // For App (argument)
}

impl TermNode {
    #[inline]
    pub fn var(index: u32) -> Self {
        TermNode {
            ty: TermType::Var,
            var_index: index,
            left: None,
            right: None,
        }
    }

    #[inline]
    pub fn abs(body: TermId) -> Self {
        TermNode {
            ty: TermType::Abs,
            var_index: 0,
            left: Some(body),
            right: None,
        }
    }

    #[inline]
    pub fn app(func: TermId, arg: TermId) -> Self {
        TermNode {
            ty: TermType::App,
            var_index: 0,
            left: Some(func),
            right: Some(arg),
        }
    }
}

/// Arena for term allocation
///
/// Provides fast allocation and bulk deallocation.
/// Terms are accessed via TermId indices.
pub struct TermArena {
    nodes: Vec<TermNode>,
}

impl TermArena {
    pub fn new() -> Self {
        TermArena {
            nodes: Vec::with_capacity(4096),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        TermArena {
            nodes: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub fn alloc(&mut self, node: TermNode) -> TermId {
        let id = self.nodes.len() as u32;
        self.nodes.push(node);
        TermId(id)
    }

    #[inline]
    pub fn get(&self, id: TermId) -> &TermNode {
        &self.nodes[id.index()]
    }

    #[inline]
    pub fn get_mut(&mut self, id: TermId) -> &mut TermNode {
        &mut self.nodes[id.index()]
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for TermArena {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level term representation (for external API)
#[derive(Debug, Clone)]
pub enum Term {
    Var(u32),
    Abs(Box<Term>),
    App(Box<Term>, Box<Term>),
}

impl Term {
    /// Convert to arena representation
    pub fn to_arena(&self, arena: &mut TermArena) -> TermId {
        match self {
            Term::Var(idx) => arena.alloc(TermNode::var(*idx)),
            Term::Abs(body) => {
                let body_id = body.to_arena(arena);
                arena.alloc(TermNode::abs(body_id))
            }
            Term::App(func, arg) => {
                let func_id = func.to_arena(arena);
                let arg_id = arg.to_arena(arena);
                arena.alloc(TermNode::app(func_id, arg_id))
            }
        }
    }

    /// Convert from arena representation
    pub fn from_arena(arena: &TermArena, id: TermId) -> Self {
        let node = arena.get(id);
        match node.ty {
            TermType::Var => Term::Var(node.var_index),
            TermType::Abs => {
                let body = Self::from_arena(arena, node.left.unwrap());
                Term::Abs(Box::new(body))
            }
            TermType::App => {
                let func = Self::from_arena(arena, node.left.unwrap());
                let arg = Self::from_arena(arena, node.right.unwrap());
                Term::App(Box::new(func), Box::new(arg))
            }
        }
    }

    /// Compute term size (number of nodes)
    pub fn size(&self) -> usize {
        match self {
            Term::Var(_) => 1,
            Term::Abs(body) => 1 + body.size(),
            Term::App(func, arg) => 1 + func.size() + arg.size(),
        }
    }

    /// Compute term depth
    pub fn depth(&self) -> usize {
        match self {
            Term::Var(_) => 1,
            Term::Abs(body) => 1 + body.depth(),
            Term::App(func, arg) => 1 + func.depth().max(arg.depth()),
        }
    }
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::Var(idx) => write!(f, "{}", idx),
            Term::Abs(body) => write!(f, "λ.{}", body),
            Term::App(func, arg) => {
                // Add parens if needed
                let func_str = match **func {
                    Term::App(_, _) => format!("({})", func),
                    _ => format!("{}", func),
                };
                let arg_str = match **arg {
                    Term::Var(_) => format!("{}", arg),
                    _ => format!("({})", arg),
                };
                write!(f, "{}{}", func_str, arg_str)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_arena() {
        let mut arena = TermArena::new();

        // Create term: λ.0
        let body = arena.alloc(TermNode::var(0));
        let abs = arena.alloc(TermNode::abs(body));

        assert_eq!(arena.len(), 2);
        assert_eq!(arena.get(abs).ty, TermType::Abs);
    }

    #[test]
    fn test_term_conversion() {
        // Term: (λ.0)(λ.0)
        let id = Term::Abs(Box::new(Term::Var(0)));
        let term = Term::App(Box::new(id.clone()), Box::new(id));

        let mut arena = TermArena::new();
        let id = term.to_arena(&mut arena);
        let reconstructed = Term::from_arena(&arena, id);

        assert_eq!(term.size(), reconstructed.size());
        assert_eq!(term.depth(), reconstructed.depth());
    }
}
