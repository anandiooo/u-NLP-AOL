import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from transformers import AutoTokenizer

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.model import load_model, build_explainer

logger = logging.getLogger(__name__)

@dataclass
class FallacyResult:
    text: str
    context: Optional[str]
    detected_fallacies: List[str]
    confidence_scores: Dict[str, float]
    explanation: str
    explainer_name: str
    has_fallacy: bool = field(init=False)

    def __post_init__(self):
        self.has_fallacy = bool(
            self.detected_fallacies
            and not (len(self.detected_fallacies) == 1 and self.detected_fallacies[0] == "no_fallacy")
        )

    def to_dict(self):
        return {
            "text": self.text,
            "context": self.context,
            "has_fallacy": self.has_fallacy,
            "detected_fallacies": self.detected_fallacies,
            "confidence_scores": self.confidence_scores,
            "explanation": self.explanation,
            "explainer_name": self.explainer_name,
        }

    def __repr__(self):
        labels = ", ".join(self.detected_fallacies) or "none"
        return (
            f"[LogiCheck Result]\n"
            f"  Text      : {self.text[:80]}{'...' if len(self.text) > 80 else ''}\n"
            f"  Fallacies : {labels}\n"
            f"  Explanation: {self.explanation}\n"
        )

class LogiCheckPipeline:
    def __init__(self, model, tokenizer, explainer, label_list, threshold, max_length, device):
        self.model = model
        self.tokenizer = tokenizer
        self.explainer = explainer
        self.label_list = label_list
        self.threshold = threshold
        self.max_length = max_length
        self.device = device
        self.model.eval()

    @classmethod
    def from_config(cls, config_path, checkpoint_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        label_list = config["labels"]

        tokenizer = AutoTokenizer.from_pretrained(config["model"]["backbone"])
        model = load_model(checkpoint_path, config, device)
        explainer = build_explainer(config)

        return cls(
            model=model,
            tokenizer=tokenizer,
            explainer=explainer,
            label_list=label_list,
            threshold=config["inference"]["threshold"],
            max_length=config["data"]["max_length"],
            device=device,
        )

    def _tokenize(self, text, context=None):
        encoding = self.tokenizer(
            text, context,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    @torch.no_grad()
    def predict(self, text, context=None, explain=True):
        encoding = self._tokenize(text, context)

        logits = self.model(
            encoding.get("input_ids"),
            encoding.get("attention_mask"),
            encoding.get("token_type_ids"),
        )

        probs = torch.softmax(logits, dim=-1).squeeze(0)
        scores = {
            self.label_list[i]: float(probs[i])
            for i in range(len(self.label_list))
        }

        pred_idx = torch.argmax(probs).item()
        detected_label = self.label_list[pred_idx]
        detected = [detected_label] if detected_label != "no_fallacy" else []

        if not detected:
            detected = ["no_fallacy"]

        explanation = ""
        if explain:
            explanation = self.explainer.explain(text, detected)

        return FallacyResult(
            text=text,
            context=context,
            detected_fallacies=detected,
            confidence_scores=scores,
            explanation=explanation,
            explainer_name=self.explainer.name
        )

    def predict_batch(self, texts, contexts=None, explain=True):
        if contexts is None:
            contexts = [None] * len(texts)
        return [self.predict(text, ctx, explain) for text, ctx in zip(texts, contexts)]
