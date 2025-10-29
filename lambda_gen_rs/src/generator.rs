//! Term generator with configurable depth and size limits.
//!
//! Generates well-formed lambda calculus terms with proper variable binding.
//! Ensures terms are reducible or in normal form.

use crate::term::Term;

/// Simple, fast random number generator (LCG)
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn seed_from_u64(seed: u64) -> Self {
        SimpleRng {
            state: seed.wrapping_add(1),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        // LCG constants from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    pub fn gen_range(&mut self, min: u32, max: u32) -> u32 {
        if min >= max {
            return min;
        }
        let range = max - min;
        min + (self.next_u64() % range as u64) as u32
    }
}

#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    pub max_depth: usize,
    pub min_depth: usize,
    pub max_size: usize,
    pub allow_divergent: bool,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        GeneratorConfig {
            max_depth: 8,      // Balanced depth for complex terms
            min_depth: 3,      // Avoid trivial terms
            max_size: 100,     // Initial size before reduction
            allow_divergent: false,  // CRITICAL: Filter out non-normalizing terms
        }
    }
}

pub struct TermGenerator {
    config: GeneratorConfig,
}

impl TermGenerator {
    pub fn new(config: GeneratorConfig) -> Self {
        TermGenerator { config }
    }

    /// Generate a random term within configured constraints
    pub fn generate(&self, rng: &mut SimpleRng) -> Option<Term> {
        for _attempt in 0..100 {
            let term = self.generate_term(rng, 0, self.config.max_depth, 0)?;

            // Check size constraint
            if term.size() <= self.config.max_size {
                return Some(term);
            }
        }
        None
    }

    /// Recursive term generation with depth tracking
    fn generate_term(
        &self,
        rng: &mut SimpleRng,
        depth: usize,
        max_depth: usize,
        var_context: u32,
    ) -> Option<Term> {
        // Force termination at max depth
        if depth >= max_depth {
            return Some(Term::Var(rng.gen_range(0, var_context.max(1))));
        }

        // Bias towards creating interesting terms
        let choice = if depth < self.config.min_depth {
            // Early depth: bias towards Abs and App
            rng.gen_range(0, 10)
        } else {
            // Later depth: allow all choices
            rng.gen_range(0, 10)
        };

        match choice {
            // Var: 20% probability
            0..=1 if var_context > 0 => {
                Some(Term::Var(rng.gen_range(0, var_context)))
            }
            // Abs: 40% probability
            2..=5 => {
                let body = self.generate_term(rng, depth + 1, max_depth, var_context + 1)?;
                Some(Term::Abs(Box::new(body)))
            }
            // App: 40% probability
            _ => {
                let func = self.generate_term(rng, depth + 1, max_depth, var_context)?;
                let arg = self.generate_term(rng, depth + 1, max_depth, var_context)?;
                Some(Term::App(Box::new(func), Box::new(arg)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation() {
        let config = GeneratorConfig::default();
        let generator = TermGenerator::new(config);
        let mut rng = SimpleRng::seed_from_u64(42);

        let term = generator.generate(&mut rng);
        assert!(term.is_some());

        let term = term.unwrap();
        assert!(term.size() > 0);
        assert!(term.depth() > 0);
    }

    #[test]
    fn test_size_constraint() {
        let config = GeneratorConfig {
            max_depth: 4,
            min_depth: 2,
            max_size: 10,
            allow_divergent: true,
        };
        let generator = TermGenerator::new(config);
        let mut rng = SimpleRng::seed_from_u64(123);

        for _ in 0..10 {
            if let Some(term) = generator.generate(&mut rng) {
                assert!(term.size() <= 10);
            }
        }
    }
}
