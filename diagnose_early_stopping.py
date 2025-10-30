#!/usr/bin/env python3
"""
Diagnose Early Stopping Issues

Analyzes why the model stops reducing terms before reaching normal form.
Shows the final term state and remaining redexes.
"""

import argparse
import json
import random
import torch

from lambda_train import LambdaSpanPredictor, LambdaTokenizer, TrainingConfig
from lambda_gen import TermGenerator, TreeReducer, Term, TermType, Renderer
from pathlib import Path


def has_redex(term: Term) -> bool:
    """Check if term contains any redex."""
    if term.type == TermType.APP and term.left and term.left.type == TermType.ABS:
        return True
    if term.type == TermType.ABS and term.body:
        return has_redex(term.body)
    elif term.type == TermType.APP:
        if term.left and has_redex(term.left):
            return True
        if term.right and has_redex(term.right):
            return True
    return False


def find_redexes(term: Term, path=[]):
    """Find all redex paths in a term."""
    redexes = []

    # Check if current is a redex
    if term.type == TermType.APP and term.left and term.left.type == TermType.ABS:
        redexes.append(path.copy())

    # Recurse
    if term.type == TermType.ABS and term.body:
        redexes.extend(find_redexes(term.body, path + [0]))
    elif term.type == TermType.APP:
        if term.left:
            redexes.extend(find_redexes(term.left, path + [0]))
        if term.right:
            redexes.extend(find_redexes(term.right, path + [1]))

    return redexes


@torch.no_grad()
def predict_with_model(model, tokenizer, term, device):
    """Predict redex and get NF confidence."""
    render_result = Renderer.to_debruijn_with_spans(term)
    term_str = render_result.string

    token_ids, offsets = tokenizer.encode(term_str, add_special=True)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    outputs = model(input_ids, attention_mask)

    nf_logit = outputs['nf_logits'][0].item()
    nf_prob = torch.sigmoid(torch.tensor(nf_logit)).item()

    start_logits = outputs['start_logits'][0]
    end_logits = outputs['end_logits'][0]

    import torch.nn.functional as F
    start_probs = F.softmax(start_logits, dim=0)
    end_probs = F.softmax(end_logits, dim=0)

    start_idx = int(start_probs.argmax().item())
    end_idx = int(end_probs.argmax().item())

    start_conf = float(start_probs[start_idx].item())
    end_conf = float(end_probs[end_idx].item())

    return {
        'nf_prob': nf_prob,
        'predicts_nf': nf_prob > 0.5,
        'start_idx': start_idx,
        'end_idx': end_idx,
        'start_conf': start_conf,
        'end_conf': end_conf
    }


def diagnose_term(model, tokenizer, term, device, reducer, max_steps=100):
    """Run model reduction and diagnose where it stops."""
    steps = []
    current = term

    for step_num in range(max_steps):
        term_str = Renderer.to_debruijn_with_spans(current).string
        actual_redexes = find_redexes(current)
        prediction = predict_with_model(model, tokenizer, current, device)

        steps.append({
            'step': step_num,
            'term': term_str,
            'term_length': len(term_str),
            'actual_redexes': len(actual_redexes),
            'redex_paths': actual_redexes[:3],  # First 3
            'model_nf_prob': prediction['nf_prob'],
            'model_predicts_nf': prediction['predicts_nf'],
            'start_conf': prediction['start_conf'],
            'end_conf': prediction['end_conf']
        })

        # Check if model stops
        if prediction['predicts_nf']:
            return {
                'stopped': True,
                'reason': 'model_predicted_nf',
                'steps': steps,
                'final_term': term_str,
                'remaining_redexes': len(actual_redexes),
                'actually_nf': len(actual_redexes) == 0
            }

        # Try to reduce
        if not actual_redexes:
            return {
                'stopped': True,
                'reason': 'actually_nf',
                'steps': steps,
                'final_term': term_str,
                'remaining_redexes': 0,
                'actually_nf': True
            }

        # Use first redex path for reduction
        try:
            current = reducer._apply_reduction(current, actual_redexes[0])
        except Exception as e:
            return {
                'stopped': True,
                'reason': f'reduction_error: {e}',
                'steps': steps,
                'final_term': term_str,
                'remaining_redexes': len(actual_redexes),
                'actually_nf': False
            }

    return {
        'stopped': True,
        'reason': 'max_steps',
        'steps': steps,
        'final_term': Renderer.to_debruijn_with_spans(current).string,
        'remaining_redexes': len(find_redexes(current)),
        'actually_nf': False
    }


def main():
    parser = argparse.ArgumentParser(description='Diagnose model early stopping')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--num-terms', type=int, default=20, help='Number of terms to diagnose')
    parser.add_argument('--output', type=str, default='diagnosis.json', help='Output file')

    args = parser.parse_args()

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    train_config_raw = checkpoint.get('config')
    if train_config_raw is None:
        config_path = Path(args.checkpoint).parent / 'config.json'
        with open(config_path) as f:
            train_config = TrainingConfig(**json.load(f))
    elif isinstance(train_config_raw, dict):
        train_config = TrainingConfig(**train_config_raw)
    else:
        train_config = train_config_raw

    tokenizer = LambdaTokenizer()
    model = LambdaSpanPredictor(train_config, tokenizer.vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model: {train_config.d_model}d × {train_config.n_layers}L")

    # Generate terms
    rng = random.Random(42)
    term_gen = TermGenerator(
        rng=rng,
        max_depth=10,
        min_depth=2,
        max_size=100,
        libraries=[],
        allow_divergent=False
    )

    reducer = TreeReducer(max_steps=1000)

    print(f"\nDiagnosing {args.num_terms} terms...\n")

    results = []
    early_stops = 0
    correct_stops = 0

    for i in range(args.num_terms):
        term = term_gen.generate()
        if not term:
            continue

        diagnosis = diagnose_term(model, tokenizer, term, device, reducer)

        # Categorize
        if diagnosis['actually_nf']:
            correct_stops += 1
            category = 'correct'
        else:
            early_stops += 1
            category = 'early_stop'

        results.append({
            'term_id': i,
            'initial_term': Renderer.to_debruijn_with_spans(term).string,
            'category': category,
            'diagnosis': diagnosis
        })

        if not diagnosis['actually_nf']:
            print(f"\n{'='*80}")
            print(f"EARLY STOP #{early_stops}: Term {i}")
            print(f"{'='*80}")
            print(f"Initial: {diagnosis['steps'][0]['term'][:80]}...")
            print(f"Final:   {diagnosis['final_term'][:80]}...")
            print(f"Remaining redexes: {diagnosis['remaining_redexes']}")
            print(f"Steps taken: {len(diagnosis['steps'])}")
            print(f"\nFinal step analysis:")
            final_step = diagnosis['steps'][-1]
            print(f"  Model NF probability: {final_step['model_nf_prob']:.4f}")
            print(f"  Actual redexes: {final_step['actual_redexes']}")
            print(f"  Redex paths: {final_step['redex_paths']}")
            print(f"  Start confidence: {final_step['start_conf']:.4f}")
            print(f"  End confidence: {final_step['end_conf']:.4f}")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total terms: {len(results)}")
    print(f"Correct stops (NF): {correct_stops} ({100*correct_stops/len(results):.1f}%)")
    print(f"Early stops: {early_stops} ({100*early_stops/len(results):.1f}%)")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == '__main__':
    main()
