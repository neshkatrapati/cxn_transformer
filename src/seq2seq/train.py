import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from .data import build_dataloader
from .model import Seq2SeqTransformer
from ..model import generate_square_subsequent_mask
import tqdm

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_inp.size(1)).to(device)
            src_pad_mask = src == 0
            tgt_pad_mask = tgt_inp == 0
            out = model(src, tgt_inp, tgt_mask=tgt_mask,
                        src_padding_mask=src_pad_mask,
                        tgt_padding_mask=tgt_pad_mask)
            loss = criterion(out.reshape(-1, out.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item() * src.size(0)
    return total_loss / len(loader.dataset)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, train_loader = build_dataloader(
        args.src, args.tgt, args.batch_size, args.min_freq, shuffle=True)
    val_loader = None
    if args.eval_src and args.eval_tgt:
        _, val_loader = build_dataloader(
            args.eval_src, args.eval_tgt, args.batch_size, args.min_freq)
    model = Seq2SeqTransformer(
        len(dataset.src_vocab), len(dataset.tgt_vocab),
        args.d_model, args.nhead, args.num_layers,
        args.dim_ff, args.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tgt_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for src, tgt in tqdm.tqdm(train_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_inp.size(1)).to(device)
            src_pad_mask = src == 0
            tgt_pad_mask = tgt_inp == 0
            optimizer.zero_grad()
            out = model(src, tgt_inp, tgt_mask=tgt_mask,
                        src_padding_mask=src_pad_mask,
                        tgt_padding_mask=tgt_pad_mask)
            loss = criterion(out.reshape(-1, out.size(-1)), tgt_out.reshape(-1))
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
    torch.save({'model_state_dict': model.state_dict(),
                'src_vocab': dataset.src_vocab,
                'tgt_vocab': dataset.tgt_vocab}, args.output)
    print('Training completed. Model saved to', args.output)

def main():
    parser = argparse.ArgumentParser(description='Train seq2seq Transformer')
    parser.add_argument('--src', required=True, help='Path to source text file')
    parser.add_argument('--tgt', required=True, help='Path to target text file')
    parser.add_argument('--eval-src', type=str, default=None)
    parser.add_argument('--eval-tgt', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-freq', type=int, default=1)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dim-ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='seq2seq_model.pt')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
