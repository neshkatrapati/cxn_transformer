#!/usr/bin/env python3
# save as seq2seq_hf.py

import argparse
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

def load_parallel(src_path, tgt_path):
    with open(src_path, encoding='utf-8') as fs, open(tgt_path, encoding='utf-8') as ft:
        src_lines = [l.strip() for l in fs if l.strip()]
        tgt_lines = [l.strip() for l in ft if l.strip()]
    assert len(src_lines) == len(tgt_lines)
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
    parser = argparse.ArgumentParser(description="Train / Eval a T5-style seq2seq on parallel files")
    parser.add_argument("--src_train", help="train source file")
    parser.add_argument("--tgt_train", help="train target file")
    parser.add_argument("--src_test",  help="test source file")
    parser.add_argument("--tgt_test",  help="test target file")
    parser.add_argument("--model_name", default="t5-small", help="HF model name or path")
    parser.add_argument("--model_path", help="If --do_eval only, path to a saved checkpoint")
    parser.add_argument("--output_dir", default="./outputs", help="where to save/train model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--max_len",    type=int, default=64)
    parser.add_argument("--do_train", action="store_true", help="Only run training")
    parser.add_argument("--do_eval",  action="store_true", help="Only run evaluation")
    args = parser.parse_args()

    # 1. prepare raw datasets if training or both
    if args.do_train or (not args.do_train and not args.do_eval):
        train_dict = load_parallel(args.src_train, args.tgt_train)
        test_dict  = load_parallel(args.src_test,  args.tgt_test)
        raw_datasets = DatasetDict({
            "train": Dataset.from_dict(train_dict),
            "test":  Dataset.from_dict(test_dict ),
        })

    # 2. load tokenizer & model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_path if args.do_eval and args.model_path else args.model_name
    )

    # 3. tokenize if training or both
    if args.do_train or (not args.do_train and not args.do_eval):
        tokenized = raw_datasets.map(
            lambda b: preprocess_batch(b, tokenizer, args.max_len),
            batched=True,
            remove_columns=["src", "tgt"],
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            save_total_limit=2,
            predict_with_generate=True,
            logging_dir=f"{args.output_dir}/logs",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset= tokenized["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics(p, tokenizer),
        )

    # 4. run training
    if args.do_train or (not args.do_train and not args.do_eval):
        trainer.train()
        trainer.save_model(args.output_dir)

    # 5. run evaluation
    if args.do_eval or (not args.do_train and not args.do_eval):
        # if we just trained, `model` is updated in-place; otherwise it was loaded from --model_path
        metrics = trainer.evaluate() if not args.do_eval else trainer.evaluate(eval_dataset=trainer.eval_dataset)
        print("Eval metrics:", metrics)

        preds = trainer.predict(tokenized["test"])
        print("Test metrics:", preds.metrics)

        # print first 5 examples
        decoded = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
        for src, tgt, pr in zip(raw_datasets["test"]["src"][:5],
                                 raw_datasets["test"]["tgt"][:5],
                                 decoded[:5]):
            print(f"\nINPUT   --> {src}")
            print(f"TARGET  --> {tgt}")
            print(f"PREDICT --> {pr}")

if __name__ == "__main__":
    main()
