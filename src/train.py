import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from data import build_dataloader
from model import TransformerLM, generate_square_subsequent_mask
import tqdm

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in data_loader:
            # src, tgt: [batch, seq]
            src = src.to(device)
            tgt = tgt.to(device)
            seq_len = src.size(1)
            # dynamic mask matching batch sequence length
            mask = generate_square_subsequent_mask(seq_len).to(device)

            output = model(src, mask)
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt.view(-1)
            )
            total_loss += loss.item() * src.size(0)
    return total_loss / len(data_loader.dataset)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build data loaders
    dataset, train_loader = build_dataloader(
        args.corpus, args.seq_len, args.batch_size
    )
    val_loader = None
    if args.eval_corpus:
        _, val_loader = build_dataloader(
            args.eval_corpus, args.seq_len, args.batch_size
        )

    vocab_size = len(dataset.vocab)
    
    print(f"Loaded dataset with vocab : {vocab}")
    model = TransformerLM(
        vocab_size,
        args.d_model,
        args.nhead,
        args.num_layers,
        args.dim_ff,
        args.dropout
    ).to(device)

    # ignore padding index
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for src, tgt in tqdm.tqdm(train_loader):
            # src, tgt: [batch, seq]
            src = src.to(device)
            tgt = tgt.to(device)
            seq_len = src.size(1)

            # generate mask for this batch
            mask = generate_square_subsequent_mask(seq_len).to(device)

            optimizer.zero_grad()
            output = model(src, mask)
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt.view(-1)
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * src.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        ppl = math.exp(avg_loss)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} ppl={ppl:.4f}")

        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, device)
            val_ppl = math.exp(val_loss)
            print(f"  Val : loss={val_loss:.4f} ppl={val_ppl:.4f}")

    torch.save(
        {'model_state_dict': model.state_dict(), 'vocab': dataset.vocab},
        args.output
    )
    print('Training completed. Model saved to', args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Transformer-based Language Model'
    )
    parser.add_argument(
        '--corpus', type=str, required=True,
        help='Path to training corpus text file'
    )
    parser.add_argument(
        '--eval-corpus', type=str, default=None,
        help='Path to validation corpus text file'
    )
    parser.add_argument(
        '--seq-len', type=int, default=32,
        help='Sequence length'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--d-model', type=int, default=128,
        help='Model hidden dimension'
    )
    parser.add_argument(
        '--nhead', type=int, default=4,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--num-layers', type=int, default=2,
        help='Number of Transformer layers'
    )
    parser.add_argument(
        '--dim-ff', type=int, default=512,
        help='Feedforward layer dimension'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1,
        help='Dropout rate'
    )
    parser.add_argument(
        '--output', type=str, default='model.pt',
        help='Output path for the saved model'
    )
    args = parser.parse_args()
    train(args)
