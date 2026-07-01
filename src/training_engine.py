import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

# implements focal loss for imbalanced data
class FocalLoss(nn.Module):
    # initialises the class instance
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.register_buffer("alpha", torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))

    # performs forward pass on input
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        probs = F.softmax(logits, dim=-1)
        p_t = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# creates focal loss from config
def build_focal_loss(config, label_list):
    label_weights = config.get("label_weights", {})
    alpha = [label_weights.get(label, 1.0) for label in label_list]
    gamma = config.get("focal_loss", {}).get("gamma", 2.0)
    return FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")

# calculates evaluation metrics
def compute_metrics(logits, targets, threshold=0.5, label_names=None):
    preds = torch.argmax(logits, dim=-1).numpy()
    y_true = targets.numpy()

    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, preds, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, preds, average="micro", zero_division=0)

    metrics = {
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "micro_f1": float(micro_f1),
    }

    if label_names:
        per_class_f1 = f1_score(y_true, preds, average=None, zero_division=0)
        per_class_p = precision_score(y_true, preds, average=None, zero_division=0)
        per_class_r = recall_score(y_true, preds, average=None, zero_division=0)
        support = np.bincount(y_true, minlength=len(label_names))
        per_class = {}
        for i, name in enumerate(label_names):
            per_class[name] = {
                "f1": float(per_class_f1[i]),
                "precision": float(per_class_p[i]),
                "recall": float(per_class_r[i]),
                "support": int(support[i]),
            }
        metrics["per_class"] = per_class

    return metrics

# prints scikit learn classification report
def print_classification_report(logits, targets, label_names, threshold=0.5):
    preds = torch.argmax(logits, dim=-1).numpy()
    y_true = targets.numpy()
    report = classification_report(y_true, preds, target_names=label_names, zero_division=0)
    print(report)
    return report

# handles the model training loop
class Trainer:
    # initialises the class instance
    def __init__(self, model, train_loader, val_loader, criterion, config, device, output_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        tc = config["training"]
        self.epochs = tc["epochs"]
        self.grad_accum = tc["gradient_accumulation_steps"]
        self.max_grad_norm = tc["max_grad_norm"]
        self.fp16 = tc["fp16"] and device.type == "cuda"
        self.threshold = config["inference"]["threshold"]

        encoder_params = list(self.model.encoder.parameters())
        head_params = list(self.model.head.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": tc["learning_rate"]},
                {"params": head_params, "lr": tc["learning_rate"] * 5},
            ],
            weight_decay=tc["weight_decay"],
        )

        total_steps = (len(train_loader) // self.grad_accum) * self.epochs
        warmup_steps = int(total_steps * tc["warmup_ratio"])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)

        self.scaler = GradScaler("cuda", enabled=self.fp16)
        self.writer = SummaryWriter(log_dir=config.get("project", {}).get("log_dir", "logs/"))
        self.best_macro_f1 = 0.0
        self.global_step = 0

    # trains model for one epoch
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            labels = batch["labels"].to(self.device)

            use_autocast = self.fp16 and self.device.type == "cuda"
            with autocast("cuda", enabled=use_autocast):
                logits = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.criterion(logits, labels)
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % 50 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar("train/loss", loss.item() * self.grad_accum, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    logger.info(
                        f"Epoch {epoch} | Step {self.global_step} | "
                        f"Loss: {loss.item() * self.grad_accum:.4f} | LR: {lr:.2e}"
                    )

            total_loss += loss.item() * self.grad_accum

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    # evaluates model on validation set
    def _evaluate(self):
        self.model.eval()
        all_logits, all_labels = [], []

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            use_autocast = self.fp16 and self.device.type == "cuda"
            with autocast("cuda", enabled=use_autocast):
                logits = self.model(input_ids, attention_mask, token_type_ids)

            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        return compute_metrics(logits, labels, self.threshold)

    # saves model checkpoint
    def _save_checkpoint(self, epoch, metrics):
        path = os.path.join(self.output_dir, "best_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        logger.info(f"Done: Saved best checkpoint -> {path} (Macro-F1: {metrics['macro_f1']:.4f})")

    # runs the full training loop
    def train(self):
        logger.info(f"Starting training: {self.epochs} epochs | fp16={self.fp16} | grad_accum={self.grad_accum}")

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(epoch)
            metrics = self._evaluate()

            logger.info(
                f"\n{'='*60}\n"
                f"Epoch {epoch}/{self.epochs}\n"
                f"  Train Loss : {train_loss:.4f}\n"
                f"  Macro-F1   : {metrics['macro_f1']:.4f}\n"
                f"  Macro-P    : {metrics['macro_precision']:.4f}\n"
                f"  Macro-R    : {metrics['macro_recall']:.4f}\n"
                f"{'='*60}"
            )

            self.writer.add_scalar("val/macro_f1", metrics["macro_f1"], epoch)
            self.writer.add_scalar("val/macro_precision", metrics["macro_precision"], epoch)
            self.writer.add_scalar("val/train_loss", train_loss, epoch)

            if metrics["macro_f1"] > self.best_macro_f1:
                self.best_macro_f1 = metrics["macro_f1"]
                self._save_checkpoint(epoch, metrics)

        self.writer.close()
        logger.info(f"Training complete. Best Macro-F1: {self.best_macro_f1:.4f}")
