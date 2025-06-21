import argparse
import torch
from .model import Seq2SeqTransformer
from .data import build_dataloader, collate_fn
from ..model import generate_square_subsequent_mask


def load_model(path, device, d_model, nhead, num_layers, dim_ff, dropout):
    checkpoint = torch.load(path, map_location=device)
    model = Seq2SeqTransformer(
        len(checkpoint['src_vocab']),
        len(checkpoint['tgt_vocab']),
        d_model, nhead, num_layers, dim_ff, dropout
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['src_vocab'], checkpoint['tgt_vocab']


def greedy_decode(model, src, src_vocab, tgt_vocab, device, max_len=50):
    src = torch.tensor([src], device=device)
    src_mask = None
    memory = model.encode(src, src_mask, src == 0)
    ys = torch.tensor([[tgt_vocab['<bos>']]], device=device)
    for _ in range(max_len):
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
        out = model.decode(ys, memory, tgt_mask=tgt_mask,
                           memory_padding_mask=src == 0,
                           tgt_padding_mask=ys == 0)
        prob = model.fc_out(out[:, -1])
        next_word = prob.argmax(dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
        if next_word == tgt_vocab['<eos>']:
            break
    return ys.squeeze(0).tolist()[1:]


def compute_accuracy(model, dataset, device):
    correct = 0
    total = 0
    for src_ids, tgt_ids in dataset.data:
        pred = greedy_decode(model, src_ids, dataset.src_vocab, dataset.tgt_vocab, device, max_len=len(tgt_ids)+2)
        # remove eos if present
        if pred and pred[-1] == dataset.tgt_vocab['<eos>']:
            pred = pred[:-1]
        target = tgt_ids[1:]  # skip bos
        length = min(len(pred), len(target))
        for p, t in zip(pred[:length], target[:length]):
            if p == t:
                correct += 1
        total += len(target)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Evaluate seq2seq model')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--src', required=True, help='Path to test source file')
    parser.add_argument('--tgt', required=True, help='Path to test target file')
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dim-ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, src_vocab, tgt_vocab = load_model(
        args.model, device, args.d_model, args.nhead, args.num_layers, args.dim_ff, args.dropout
    )
    dataset, _ = build_dataloader(args.src, args.tgt, batch_size=1, shuffle=False, min_freq=1)
    dataset.src_vocab = src_vocab
    dataset.tgt_vocab = tgt_vocab
    acc = compute_accuracy(model, dataset, device)
    print(f'Accuracy: {acc*100:.2f}%')


if __name__ == '__main__':
    main()
