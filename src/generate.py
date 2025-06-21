import argparse
import torch
from model import TransformerLM, generate_square_subsequent_mask


def load_model(model_path, d_model, nhead, num_layers, dim_ff, dropout, device):
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint['vocab']
    model = TransformerLM(
        len(vocab), d_model, nhead, num_layers, dim_ff, dropout
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    inv_vocab = {i: t for t, i in vocab.items()}
    return model, vocab, inv_vocab


def generate(model, vocab, inv_vocab, prompt, length, temperature, device):
    tokens = prompt.split()
    if tokens:
        ids = [vocab.get(t, vocab['<unk>']) for t in tokens]
    else:
        # choose a random non-special token to start
        specials = {vocab['<pad>'], vocab['<unk>']}
        choices = [i for i in range(len(vocab)) if i not in specials]
        ids = [choices[0] if choices else vocab['<unk>']]

    for _ in range(length):
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        mask = generate_square_subsequent_mask(inp.size(1)).to(device)
        with torch.no_grad():
            out = model(inp, mask)
        next_token_logits = out[0, -1] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)

    return " ".join(inv_vocab.get(i, "<unk>") for i in ids)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument('--model', required=True, help='Path to the saved model file')
    parser.add_argument('--prompt', type=str, default='', help='Seed text to start generation')
    parser.add_argument('--length', type=int, default=20, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    # architecture parameters (should match training)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dim-ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, vocab, inv_vocab = load_model(
        args.model, args.d_model, args.nhead, args.num_layers, args.dim_ff, args.dropout, device
    )
    text = generate(model, vocab, inv_vocab, args.prompt, args.length, args.temperature, device)
    print(text)


if __name__ == '__main__':
    main()
