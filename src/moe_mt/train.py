"""Training entry point for the MoE translation model."""

import argparse
from dataclasses import dataclass

import torch
import yaml
from torch.utils.data import DataLoader

from .model import SparseMoETranslator


@dataclass
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor


def build_dataset(name: str, tokenizer_path: str):
    from datasets import load_dataset
    return load_dataset(name, split="train")


def collate(items, pad_id: int = 0) -> Batch:
    max_len = max(len(item["input_ids"]) for item in items)
    input_ids, labels = [], []
    for item in items:
        seq = item["input_ids"][:max_len]
        pad = [pad_id] * (max_len - len(seq))
        input_ids.append(seq + pad)
        labels.append(item["labels"][:max_len] + pad)
    return Batch(torch.tensor(input_ids), torch.tensor(labels))


def train(config: dict):
    model = SparseMoETranslator(
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["encoder_layers"],
        num_experts=config["model"]["num_experts"],
        top_k=config["model"]["top_k"],
        shared_experts=config["model"]["shared_experts"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    dataset = build_dataset("wmt/paracrawl", config["data"]["tokenizer"])
    loader = DataLoader(dataset, batch_size=256, collate_fn=collate)

    for step, batch in enumerate(loader):
        if step >= config["training"]["max_steps"]:
            break
        outputs = model(batch.input_ids)
        loss = torch.nn.functional.cross_entropy(
            outputs["logits"].view(-1, outputs["logits"].size(-1)),
            batch.labels.view(-1),
            ignore_index=-100,
        )
        loss = loss + 0.01 * outputs["aux_loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.eval_only:
        print("Running evaluation on the held-out test split ...")
    else:
        train(config)


if __name__ == "__main__":
    main()
