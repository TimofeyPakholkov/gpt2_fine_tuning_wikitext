import argparse

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          default_data_collator,
                          Trainer, TrainingArguments
                         )
from datasets import load_dataset
from itertools import chain
import torch
import numpy as np
import evaluate


class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


def compute_metrics(eval_pred):
    bleu = evaluate.load("bleu")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return bleu.compute(predictions=predictions, references=labels)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = model.config.eos_token_id

    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", args.dataset)
    column_names_train = list(dataset['train'].features)
    column_names_val = list(dataset['test'].features)

    train = dataset['train']
    val = dataset['test']


    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True)
    

    def group_texts(examples):
        block_size = tokenizer.model_max_length
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    train = train.map(preprocess, remove_columns=column_names_train, batched=True)
    train = train.map(group_texts, batched=True)
    val = val.map(preprocess, remove_columns=column_names_val, batched=True)
    val = val.map(group_texts, batched=True)

    training_args = TrainingArguments(
            output_dir=args.output_folder,
            logging_dir=args.logging_folder,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=4,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='wikitext-2-v1')
    parser.add_argument("--model_name", type=str, default='openai-community/gpt2')
    parser.add_argument("--output_folder", type=str, default='./output')
    parser.add_argument("--logging_folder", type=str, default='./logs')
    args = parser.parse_args()
    main(args)
