import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

DATASET_SCHEMAS: Dict[str, Dict] = {
    "cocolo_fa": {
        "text_col": "comment",
        "label_col": "fallacy_type",
        "context_col": None,
    },
    "logic": {
        "text_col": "statement",
        "label_col": "label",
        "context_col": None,
    },
    "counterfactual": {
        "text_col": "fallacious_text",
        "label_col": "fallacy",
        "context_col": "corrected_text",
    },
    "ptc": {
        "text_col": "text",
        "labels_col": "techniques",
        "context_col": None,
    },
}

LABEL_MAP = {
    "appeal to authority": "appeal_to_authority",
    "appeal to majority": "appeal_to_majority",
    "appeal to nature": "appeal_to_nature",
    "appeal to tradition": "appeal_to_tradition",
    "appeal to worse problems": "appeal_to_worse_problems",
    "appeal to worse problem": "appeal_to_worse_problems",
    "false dilemma": "false_dilemma",
    "hasty generalization": "hasty_generalization",
    "faulty generalization": "hasty_generalization",
    "slippery slope": "slippery_slope",
    "no fallacy": "no_fallacy",
    "valid": "no_fallacy",
    "none": "no_fallacy",
}

def _normalise_label(raw):
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip().lower()
    if cleaned in LABEL_MAP:
        return LABEL_MAP[cleaned]
    if cleaned in LABEL_MAP.values():
        return cleaned
    alt = cleaned.replace("_", " ")
    if alt in LABEL_MAP:
        return LABEL_MAP[alt]
    return None

def _parse_labels(raw):
    if isinstance(raw, list):
        labels = raw
    elif isinstance(raw, str):
        try:
            labels = json.loads(raw) if raw.startswith("[") else [x.strip() for x in raw.split(",")]
        except (json.JSONDecodeError, ValueError):
            labels = [raw]
    elif isinstance(raw, dict):
        labels = [str(raw)]
    else:
        labels = [str(raw)]
    canonical = [_normalise_label(l) for l in labels]
    return [l for l in canonical if l is not None]

def load_csv(path: str, schema_key: str = ""):
    schema = DATASET_SCHEMAS.get(schema_key, {})
    df = pd.read_csv(path)

    text_col = schema.get("text_col", "text")
    context_col = schema.get("context_col")

    if "labels_col" in schema:
        label_src = schema["labels_col"]
        df["labels"] = df[label_src].apply(_parse_labels)
    else:
        label_src = schema.get("label_col", "label")
        df["labels"] = df[label_src].apply(lambda x: _parse_labels(str(x)))

    df["text"] = df[text_col].astype(str)
    df["context"] = df[context_col].astype(str) if context_col and context_col in df else None

    return df[["text", "context", "labels"]].dropna(subset=["text"])

def load_hf_dataset(dataset_name: str, split: str = "train"):
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    df = ds.to_pandas()
    logger.info(f"HF dataset '{dataset_name}' columns: {list(df.columns)}")
    return df

def load_json(path, schema_key=""):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return pd.DataFrame(columns=["text", "context", "labels"])
        if content.startswith("["):
            records = json.loads(content)
        else:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]

    raw_df = pd.DataFrame(records)
    logger.info(f"Loaded {len(raw_df)} raw records from {path}")

    if "comments" in raw_df.columns:
        logger.info(f"Detected nested comments in {path} - flattening...")
        flattened = []
        for _, article in raw_df.iterrows():
            article_context = article.get("content", article.get("title", ""))
            comments = article.get("comments", [])
            if not isinstance(comments, list):
                continue
            for c in comments:
                if not isinstance(c, dict):
                    continue
                text = c.get("comment", "")
                label_raw = c.get("fallacy", "no_fallacy")
                labels = _parse_labels(label_raw)
                if labels:
                    flattened.append({
                        "text": str(text),
                        "context": str(article_context),
                        "labels": labels,
                    })
        df = pd.DataFrame(flattened)
    else:
        df = raw_df.copy()
        text_col = "text" if "text" in df.columns else df.columns[0]
        label_col = "labels" if "labels" in df.columns else df.columns[-1]
        df["text"] = df[text_col].astype(str)
        df["labels"] = df[label_col].apply(_parse_labels)
        df["context"] = df["context"] if "context" in df.columns else None
        df = df[["text", "context", "labels"]]

    initial_len = len(df)
    df = df[df["labels"].apply(len) > 0].reset_index(drop=True)
    logger.info(f"Processed {initial_len} samples, kept {len(df)} with valid labels from {path}")
    return df

def load_all_datasets(data_dir):
    data_path = Path(data_dir)
    frames = []
    for stem in ["train", "dev", "test"]:
        candidate = data_path / f"{stem}.json"
        if candidate.exists():
            logger.info(f"Loading {candidate} as '{stem}'")
            df = load_json(str(candidate), stem)
            df["source"] = stem
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No dataset files found in {data_path}. Expected: train.json, dev.json, test.json")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["labels"].apply(len) > 0].reset_index(drop=True)
    logger.info(f"Total samples after merge: {len(combined)}")
    return combined

def load_dataset_splits(data_dir):
    data_path = Path(data_dir)
    split_files = {"train": "train", "dev": "val", "test": "test"}
    splits = {}
    for filename, split_name in split_files.items():
        candidate = data_path / f"{filename}.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Missing split file: {candidate}")
        df = load_json(str(candidate), filename)
        df["source"] = filename
        df = df[df["labels"].apply(len) > 0].reset_index(drop=True)
        splits[split_name] = df
    logger.info(
        "Loaded explicit splits -> train: %d, val: %d, test: %d",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )
    return splits


def split_dataset(df, train_ratio=0.8, val_ratio=0.1, seed=42):
    df["_strat"] = df["labels"].apply(lambda x: x[0] if x else "no_fallacy")
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=seed, stratify=df["_strat"])
    relative_val = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - relative_val), random_state=seed, stratify=temp_df["_strat"])
    for split in [train_df, val_df, test_df]:
        split.drop(columns=["_strat"], inplace=True)
    logger.info(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return {"train": train_df, "val": val_df, "test": test_df}

class FallacyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, label_list: List[str], max_length: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label2idx = {label: i for i, label in enumerate(label_list)}
        self.max_length = max_length
        self.num_labels = len(label_list)

    def __len__(self):
        return len(self.df)

    def _encode_label(self, labels):
        for label in labels:
            idx = self.label2idx.get(label)
            if idx is not None:
                return torch.tensor(idx, dtype=torch.long)
        return torch.tensor(self.label2idx.get("no_fallacy", self.num_labels - 1), dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])

        context = None
        if "context" in row and pd.notna(row["context"]) and str(row["context"]).strip():
            context = str(row["context"])

        encoding = self.tokenizer(
            text, context,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label_idx = self._encode_label(row["labels"])
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_idx,
        }

def build_dataloaders(splits, tokenizer, label_list, batch_size=8, max_length=256):
    loaders = {}
    for split_name, df in splits.items():
        dataset = FallacyDataset(df, tokenizer, label_list, max_length)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=4,
            pin_memory=True,
        )
    return loaders
