import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.amp import autocast
from dotenv import load_dotenv
load_dotenv()
from transformers import AutoTokenizer

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data import load_all_datasets, load_dataset_splits, split_dataset, build_dataloaders
from src.model import FallacyClassifier
from src.engine import build_focal_loss, compute_metrics, print_classification_report, Trainer
from src.utils import get_device, load_config, set_seed, setup_logging, count_parameters

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train LogiCheck fallacy classifier")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="data/raw/", help="directory with raw dataset files")
    parser.add_argument("--output-dir", default=None, help="override config output_dir")
    parser.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true", help="skip training, only run test evaluation")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    output_dir = args.output_dir or config["project"]["output_dir"]
    setup_logging(config["project"].get("log_dir", "logs/"))
    set_seed(config["project"]["seed"])
    device = get_device()

    label_list = config["labels"]
    tc = config["training"]
    dc = config["data"]

    logger.info("Loading datasets...")
    try:
        splits = load_dataset_splits(args.data_dir)
    except FileNotFoundError:
        df = load_all_datasets(args.data_dir)
        splits = split_dataset(df, train_ratio=dc["train_ratio"], val_ratio=dc["val_ratio"], seed=config["project"]["seed"])

    logger.info(f"Loading tokenizer: {config['model']['backbone']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])

    loaders = build_dataloaders(
        splits, tokenizer=tokenizer, label_list=label_list,
        batch_size=tc["batch_size"], max_length=dc["max_length"],
    )

    logger.info("Building model...")
    model = FallacyClassifier(
        model_name=config["model"]["backbone"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
        gradient_checkpointing=config["model"]["gradient_checkpointing"],
    )
    logger.info(f"Parameters: {count_parameters(model)}")

    if args.resume:
        logger.info(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    model.to(device)
    model.float()

    if not args.eval_only:
        criterion = build_focal_loss(config, label_list)
        trainer = Trainer(
            model=model, train_loader=loaders["train"], val_loader=loaders["val"],
            criterion=criterion, config=config, device=device, output_dir=output_dir,
        )
        trainer.train()

        best_ckpt = f"{output_dir}/best_model.pt"
        logger.info(f"Loading best checkpoint for test eval: {best_ckpt}")
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    logger.info("\n" + "=" * 60 + "\nFINAL TEST SET EVALUATION\n" + "=" * 60)
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in loaders["test"]:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            use_autocast = device.type == "cuda"
            with autocast("cuda", enabled=use_autocast):
                logits = model(input_ids, attention_mask, token_type_ids)
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())

    logits_all = torch.cat(all_logits)
    labels_all = torch.cat(all_labels)

    metrics = compute_metrics(logits_all, labels_all, threshold=config["inference"]["threshold"], label_names=label_list)

    logger.info(f"Test Macro-F1 : {metrics['macro_f1']:.4f}")
    logger.info(f"Test Macro-P  : {metrics['macro_precision']:.4f}")
    logger.info(f"Test Macro-R  : {metrics['macro_recall']:.4f}")
    logger.info(f"Test Micro-F1 : {metrics['micro_f1']:.4f}")

    print("\n--- Per-class metrics ---")
    print_classification_report(logits_all, labels_all, label_list, config["inference"]["threshold"])

if __name__ == "__main__":
    main()
