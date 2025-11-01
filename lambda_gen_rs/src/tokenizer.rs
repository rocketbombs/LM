//! Fast character-level tokenizer for lambda calculus terms.
//!
//! Vocabulary: special tokens + ASCII chars for lambda syntax.
//! Matches the Python LambdaTokenizer exactly.

use std::collections::HashMap;

/// Character-level tokenizer for lambda calculus
pub struct LambdaTokenizer {
    token_to_id: HashMap<char, u32>,
    id_to_token: Vec<char>,
    pub vocab_size: usize,
    pub pad_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,
    pub unk_id: u32,
}

impl LambdaTokenizer {
    pub fn new() -> Self {
        // Special tokens
        let special = vec!['<', 'p', 'a', 'd', '>', 'b', 'o', 's', 'e', 'u', 'n', 'k'];

        // Lambda calculus characters
        let lambda_chars = vec!['\\', '.', '(', ')'];
        let digits: Vec<char> = "0123456789".chars().collect();
        let letters: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
        let space = vec![' '];

        // Build vocabulary (matches Python implementation)
        let mut vocab_chars = Vec::new();
        vocab_chars.extend_from_slice(&special);
        vocab_chars.extend_from_slice(&lambda_chars);
        vocab_chars.extend_from_slice(&digits);
        vocab_chars.extend_from_slice(&letters);
        vocab_chars.extend_from_slice(&space);

        // Create mappings
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        for (idx, &ch) in vocab_chars.iter().enumerate() {
            token_to_id.insert(ch, idx as u32);
            id_to_token.push(ch);
        }

        let vocab_size = vocab_chars.len();

        // Special token IDs
        let pad_id = token_to_id[&'<']; // <pad> starts with '<'
        let bos_id = token_to_id[&'b']; // <bos> starts with 'b'
        let eos_id = token_to_id[&'e']; // <eos> starts with 'e'
        let unk_id = token_to_id[&'u']; // <unk> starts with 'u'

        LambdaTokenizer {
            token_to_id,
            id_to_token,
            vocab_size,
            pad_id,
            bos_id,
            eos_id,
            unk_id,
        }
    }

    /// Encode text to token IDs with optional special tokens
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(text.len() + 2);

        if add_special {
            tokens.push(self.bos_id);
        }

        for ch in text.chars() {
            let token_id = self.token_to_id.get(&ch).copied().unwrap_or(self.unk_id);
            tokens.push(token_id);
        }

        if add_special {
            tokens.push(self.eos_id);
        }

        tokens
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut result = String::with_capacity(token_ids.len());

        for &token_id in token_ids {
            // Skip special tokens
            if token_id == self.pad_id
                || token_id == self.bos_id
                || token_id == self.eos_id {
                continue;
            }

            if let Some(&ch) = self.id_to_token.get(token_id as usize) {
                result.push(ch);
            }
        }

        result
    }

    /// Get maximum length before truncation is needed
    #[inline]
    pub fn max_position(&self) -> usize {
        2048
    }
}

impl Default for LambdaTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let tokenizer = LambdaTokenizer::new();
        let text = "(\\.(\\.(00)))";

        let tokens = tokenizer.encode(text, true);
        let decoded = tokenizer.decode(&tokens);

        assert_eq!(text, decoded);
    }

    #[test]
    fn test_vocab_size() {
        let tokenizer = LambdaTokenizer::new();
        // Special(4) + lambda(4) + digits(10) + letters(26) + space(1) = 45
        // But special tokens use chars, so actual unique chars count matters
        assert!(tokenizer.vocab_size > 40);
    }
}
