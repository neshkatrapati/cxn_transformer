#!/usr/bin/env python3
import argparse
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

def load_parallel(src_path, tgt_path):
    with open(src_path, encoding='utf-8') as fs, open(tgt_path, encoding='utf-8') as ft:
        src_lines = [l.strip() for l in fs if l.strip()]
        tgt_lines = [l.strip() for l in ft if l.strip()]
    assert len(src_lines) == len(tgt_lines), "Mismatched lines in src/tgt"
    return {"src": src_lines, "tgt": tgt_lines}

def preprocess_batch(batch, tokenizer, max_length):
    inputs = tokenizer(batch["src"], max_length=max_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["tgt"], max_length=max_length, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    acc = np.mean([int(p.strip() == l.strip()) for p, l in zip(decoded_preds, decoded_labels)])
    return {"exact_match_accuracy": acc}

def main():
    parser = argparse.ArgumentParser(description="Train / Eval a T5 seq2seq model")
    parser.add_argument("--src_train", help="Path to training source file")
    parser.add_argument("--tgt_train", help="Path to training target file")
    parser.add_argument("--src_test",  help="Path to test source file")
    parser.add_argument("--tgt_test",  help="Path to test target file")
    parser.add_argument("--model_name", default="t5-small", help="HF model name or checkpoint")
    parser.add_argument("--model_path", help="Path to saved model for eval-only")
    parser.add_argument("--output_dir", default="./outputs", help="Save/fetch model here")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--max_len",    type=int, default=64)
    parser.add_argument("--do_train", action="store_true", help="Run training only")
    parser.add_argument("--do_eval",  action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # TRAIN ONLY
    if args.do_train:
        # prepare datasets
        train_dict = load_parallel(args.src_train, args.tgt_train)
        test_dict  = load_parallel(args.src_test,  args.tgt_test)
        raw = DatasetDict({
            "train": Dataset.from_dict(train_dict),
            "test":  Dataset.from_dict(test_dict),
        })
        tokenized = raw.map(
            lambda b: preprocess_batch(b, tokenizer, args.max_len),
            batched=True, remove_columns=["src","tgt"]
        )

        # model & trainer
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        train_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            save_total_limit=2,
            predict_with_generate=True,
            logging_dir=f"{args.output_dir}/logs",
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=train_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["test"],
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=lambda p: compute_metrics(p, tokenizer),
        )

        trainer.train()
        trainer.save_model(args.output_dir)

    # EVAL ONLY (or after training)
    if args.do_eval:
        load_dir = args.model_path or args.output_dir
        model = T5ForConditionalGeneration.from_pretrained(load_dir)

        test_dict = load_parallel(args.src_test, args.tgt_test)
        raw_test = Dataset.from_dict(test_dict)
        tokenized_test = raw_test.map(
            lambda b: preprocess_batch(b, tokenizer, args.max_len),
            batched=True, remove_columns=["src","tgt"]
        )

        collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        eval_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=args.batch_size,
            predict_with_generate=True,
        )
        eval_trainer = Seq2SeqTrainer(
            model=model,
            args=eval_args,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=lambda p: compute_metrics(p, tokenizer),
        )

        metrics = eval_trainer.evaluate()
        print("Evaluation metrics:", metrics)

        preds = eval_trainer.predict(tokenized_test)
        print("Test metrics:", preds.metrics)
        decoded = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
        for src, tgt, pr in zip(test_dict["src"][:5], test_dict["tgt"][:5], decoded[:5]):
            print(f"\nINPUT   --> {src}")
            print(f"TARGET  --> {tgt}")
            print(f"PREDICT --> {pr}")

if __name__ == "__main__":
    main()
