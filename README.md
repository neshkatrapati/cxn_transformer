# Simple Transformer Language Model

This project provides a small transformer-based causal language model
implemented with PyTorch. The corpus should be a plain text file where
tokens are separated by spaces and sentences by new lines.
During preprocessing each line is treated independently so training
samples never cross sentence boundaries.

## Requirements

Install dependencies via
```bash
pip install -r requirements.txt
```

## Training

Run training with
```bash
python -m src.train --corpus path/to/text.txt \
    --seq-len 32 --batch-size 32 --epochs 10 \
    --d-model 128 --nhead 4 --num-layers 2
```

Model size is configurable with the `--d-model`, `--nhead` and
`--num-layers` arguments. After training, a model file containing the
weights and vocabulary will be saved to `model.pt` by default.

During training the script prints cross-entropy loss and perplexity to
measure model quality.

## Text Generation

After training you can sample text using the saved model:

```bash
python -m src.generate --model model.pt --prompt "The" --length 20
```

The architecture parameters should match those used during training and
default to the same values as in `src.train`.

## Sequence-to-Sequence Transformer

The `src/seq2seq` package contains a small transformer model for tasks with
parallel text files. Each line in the source file corresponds to a line in the
target file. Training can be started with

```bash
python -m src.seq2seq.train --src path/to/train.src --tgt path/to/train.tgt \
    --epochs 10 --batch-size 32
```
The training dataloader shuffles the data to encourage better optimisation.
Before training starts the script prints how many `<unk>` tokens appear in the
dataset so you can verify your vocabularies cover most tokens.

A trained model can be evaluated on a test set using

```bash
python -m src.seq2seq.evaluate --model seq2seq_model.pt \
    --src path/to/test.src --tgt path/to/test.tgt
```

The evaluation script reports token accuracy based on greedy decoding.
