#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
    processors,
)
from transformers import (
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "bos_token": "<bos>",
    "eos_token": "<eos>"
}

def train_whitespace_tokenizer(files, vocab_size=5000):
    # 1. set up a WordLevel (whitespace) tokenizer
    tok = Tokenizer(models.WordLevel(unk_token=SPECIAL_TOKENS["unk_token"]))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS.values())
    )
    tok.train(files, trainer)

    # 2. post-process so we get BOS/EOS around single sentences
    tok.post_processor = processors.TemplateProcessing(
        single=f"{SPECIAL_TOKENS['bos_token']} $A {SPECIAL_TOKENS['eos_token']}",
        special_tokens=[
            (SPECIAL_TOKENS["bos_token"], tok.token_to_id(SPECIAL_TOKENS["bos_token"])),
            (SPECIAL_TOKENS["eos_token"], tok.token_to_id(SPECIAL_TOKENS["eos_token"])),
        ],
    )

    # 3. wrap in the HF Fast tokenizer API
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        **SPECIAL_TOKENS
    )

def load_parallel(src_path, tgt_path):
    with open(src_path, encoding="utf-8") as fs, open(tgt_path, encoding="utf-8") as ft:
        src = [l.strip() for l in fs if l.strip()]
        tgt = [l.strip() for l in ft if l.strip()]
    assert len(src) == len(tgt)
    return {"src": src, "tgt": tgt}

def preprocess(batch, tokenizer, max_len):
    # whitespace tokenizer already spits out ID lists
    enc = tokenizer(batch["src"], truncation=True, padding="max_length", max_length=max_len)
    with tokenizer.as_target_tokenizer():
        lab = tokenizer(batch["tgt"], truncation=True, padding="max_length", max_length=max_len)
    enc["labels"] = lab["input_ids"]
    return enc

def compute_metrics(preds_and_labels, tokenizer):
    preds, labels = preds_and_labels
    dec_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # replace -100 with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # exact string match
    acc = np.mean([int(p == l) for p, l in zip(dec_preds, dec_labels)])
    return {"exact_match": acc}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_train"); p.add_argument("--tgt_train")
    p.add_argument("--src_test");  p.add_argument("--tgt_test")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--vocab_size", type=int, default=2000)
    p.add_argument("--d_model",    type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_heads",  type=int, default=4)
    p.add_argument("--dim_ff",     type=int, default=512)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--max_len",    type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs",     type=int, default=5)
    p.add_argument("--do_train",   action="store_true")
    p.add_argument("--do_eval",    action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # 1. Prepare / load tokenizer
    tokenizer_path = out_dir / "whitespace_tokenizer"
    if args.do_train:
        tok = train_whitespace_tokenizer(
            [args.src_train, args.tgt_train],
            vocab_size=args.vocab_size
        )
        tok.save_pretrained(str(tokenizer_path))
    else:
        tok = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    # 2. Build model from scratch
    config = T5Config(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        encoder_ffn_dim=args.dim_ff,
        decoder_ffn_dim=args.dim_ff,
        encoder_layers=args.num_layers,
        decoder_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
        is_encoder_decoder=True,
    )
    
    # after your T5Config(...) call
    config.pad_token_id            = tok.pad_token_id
    config.eos_token_id            = tok.eos_token_id
    config.decoder_start_token_id  = tok.pad_token_id   # T5 uses pad as the <start> symbol
    
    model = T5ForConditionalGeneration(config)

    # 3. Datasets + tokenization
    if args.do_train:
        train_dict = load_parallel(args.src_train, args.tgt_train)
        test_dict  = load_parallel(args.src_test,  args.tgt_test)
        raw = DatasetDict({
            "train": Dataset.from_dict(train_dict),
            "test" : Dataset.from_dict(test_dict),
        })
        tokenized = raw.map(
            lambda b: preprocess(b, tok, args.max_len),
            batched=True,
            remove_columns=["src","tgt"]
        )

    # 4. Trainer setup
    data_collator = DataCollatorForSeq2Seq(tok, model=model, pad_to_multiple_of=None)
    if args.do_train:
        train_args = Seq2SeqTrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            predict_with_generate=True,
            generation_max_length=args.max_len,
            save_total_limit=2,
            logging_strategy = "steps",
            logging_steps = 20,
            learning_rate = 1e-3,
            logging_dir=str(out_dir / "logs"),
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=train_args,
            train_dataset=tokenized["train"],
            eval_dataset= tokenized["test"],
            tokenizer=tok,
            data_collator=data_collator,
            compute_metrics=lambda pl: compute_metrics(pl, tok),
        )
        trainer.train()
        trainer.save_model(str(out_dir / "final-model"))

    # 5. Evaluation only
    if args.do_eval:
        # reload model if needed
        model = T5ForConditionalGeneration.from_pretrained(str(out_dir / "final-model"), config=config)
        test_dict = load_parallel(args.src_test, args.tgt_test)
        test_ds   = Dataset.from_dict(test_dict)
        token_test= test_ds.map(
            lambda b: preprocess(b, tok, args.max_len),
            batched=True,
            remove_columns=["src","tgt"]
        )
        eval_args = Seq2SeqTrainingArguments(
            output_dir=str(out_dir),
            per_device_eval_batch_size=args.batch_size,
            predict_with_generate=True,
            generation_max_length=args.max_len,
        )
        eval_trainer = Seq2SeqTrainer(
            model=model,
            args=eval_args,
            eval_dataset=token_test,
            tokenizer=tok,
            data_collator=data_collator,
            compute_metrics=lambda pl: compute_metrics(pl, tok),
        )
        metrics = eval_trainer.evaluate()
        print("Eval metrics:", metrics)
        preds = eval_trainer.predict(token_test)
        print("Test metrics:", preds.metrics)

        # show a few
        dec = tok.batch_decode(preds.predictions, skip_special_tokens=True)
        for src, tgt, pr in zip(test_dict["src"][:5], test_dict["tgt"][:5], dec[:5]):
            print(f"\nINPUT  -> {src}")
            print(f"TARGET -> {tgt}")
            print(f"PREDICT-> {pr}")

if __name__ == "__main__":
    main()
