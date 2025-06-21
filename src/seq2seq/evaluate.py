import argparse
import torch
from .model import Seq2SeqTransformer
from ..model import generate_square_subsequent_mask
import tqdm

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


def load_tokenized_dataset(src_path, tgt_path, src_vocab, tgt_vocab):
    """Load tokenized parallel data using the given vocabularies."""
    with open(src_path, 'r', encoding='utf-8') as f:
        src_lines = [l.strip().split() for l in f if l.strip()]
    with open(tgt_path, 'r', encoding='utf-8') as f:
        tgt_lines = [l.strip().split() for l in f if l.strip()]
    assert len(src_lines) == len(tgt_lines), "Source and target files must have same number of lines"

    data = []
    unk_src = 0
    unk_tgt = 0
    for s_tok, t_tok in zip(src_lines, tgt_lines):
        src_ids = []
        for tok in s_tok:
            idx = src_vocab.get(tok, src_vocab.get('<unk>', 1))
            if idx == src_vocab.get('<unk>', 1):
                unk_src += 1
            src_ids.append(idx)
        tgt_ids = [tgt_vocab.get('<bos>')]
        for tok in t_tok:
            idx = tgt_vocab.get(tok, tgt_vocab.get('<unk>', 1))
            if idx == tgt_vocab.get('<unk>', 1):
                unk_tgt += 1
            tgt_ids.append(idx)
        tgt_ids.append(tgt_vocab.get('<eos>'))
        data.append((src_ids, tgt_ids))

    class SimpleDataset:
        pass

    dataset = SimpleDataset()
    dataset.src_path = src_path
    dataset.tgt_path = tgt_path
    dataset.src_vocab = src_vocab
    dataset.tgt_vocab = tgt_vocab
    dataset.data = data
    dataset.src_unk_count = unk_src
    dataset.tgt_unk_count = unk_tgt
    return dataset


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
    svocab = {y : x for x, y in dataset.src_vocab.items()}
    
    rvocab = {y : x for x, y in dataset.tgt_vocab.items()}
    for src_ids, tgt_ids in tqdm.tqdm(dataset.data):
        pred = greedy_decode(model, src_ids, dataset.src_vocab, dataset.tgt_vocab, device, max_len=len(tgt_ids)+2)
        # remove eos if present
        if pred and pred[-1] == dataset.tgt_vocab['<eos>']:
            pred = pred[:-1]
        
        print(f"Input : {[svocab[x] for x in src_ids]}")
        
        print(f"Pred : {[rvocab[x] for x in pred]}")
        target = tgt_ids[1:]  # skip bos
        print(f"Target : {[rvocab[x] for x in target]}")
        
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
    dataset = load_tokenized_dataset(args.src, args.tgt, src_vocab, tgt_vocab)
    if dataset.src_unk_count or dataset.tgt_unk_count:
        print(f"<unk> tokens in evaluation data - src: {dataset.src_unk_count}, tgt: {dataset.tgt_unk_count}")
    else:
        print("No <unk> tokens in evaluation data")
    acc = compute_accuracy(model, dataset, device)
    print(f'Accuracy: {acc*100:.2f}%')


if __name__ == '__main__':
    main()
