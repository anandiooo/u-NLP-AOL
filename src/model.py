import logging
import os
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)

FALLACY_DEFINITIONS = {
    "appeal_to_authority": "citing an authority figure as evidence without proper reasoning",
    "appeal_to_majority": "treating popularity as proof that a claim is true",
    "appeal_to_nature": "claiming something is good or right because it is natural",
    "appeal_to_tradition": "arguing that something is correct because it has always been done that way",
    "appeal_to_worse_problems": "dismissing an issue by pointing to a supposedly bigger problem",
    "false_dilemma": "presenting only two options when more alternatives exist",
    "hasty_generalization": "drawing a broad conclusion from a small or unrepresentative sample",
    "slippery_slope": "assuming a chain of events will inevitably follow without justification",
    "no_fallacy": "valid reasoning with no detected logical fallacy",
}

SYSTEM_PROMPT = (
    "You are a critical thinking educator specializing in logical fallacies. "
    "Your task is to explain logical fallacies in clear, educational language for a general audience. "
    "Be concise (2-4 sentences), specific to the given text, and non-judgmental in tone. "
    "Always explain WHAT the fallacy is and HOW it appears in the specific text."
)

class FallacyClassifier(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-small", num_labels=9, dropout=0.2, gradient_checkpointing=False):
        super().__init__()
        self.num_labels = num_labels

        hf_config = AutoConfig.from_pretrained(model_name)
        hf_config.hidden_dropout_prob = dropout
        hf_config.attention_probs_dropout_prob = dropout
        self.encoder = AutoModel.from_pretrained(model_name, config=hf_config, torch_dtype=torch.float32)

        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        hidden_size = hf_config.hidden_size

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_repr)

def load_model(checkpoint_path, config, device):
    model = FallacyClassifier(
        model_name=config["model"]["backbone"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
        gradient_checkpointing=False,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model

class BaseExplainer(ABC):
    @property
    @abstractmethod
    def name(self):
        ...

    @abstractmethod
    def explain(self, text, fallacy_labels):
        ...

class TemplateExplainer(BaseExplainer):
    @property
    def name(self):
        return "Local Template"

    def explain(self, text, fallacy_labels):
        labels = [l for l in fallacy_labels if l != "no_fallacy"]
        if not labels:
            return "No logical fallacy detected. The reasoning appears valid."
        parts = []
        for label in labels:
            name = label.replace("_", " ").title()
            definition = FALLACY_DEFINITIONS.get(label, "a logical error")
            parts.append(f"This text employs a **{name}**, which involves {definition}.")
        return " ".join(parts)

class GeminiExplainer(BaseExplainer):
    def __init__(self, model_name="gemini-3.1-pro", max_tokens=256, temperature=0.3):
        self.model_name = model_name
        self._name = f"Gemini {model_name.split('-')[1]}" if "-" in model_name else "Gemini"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        self._fallback = TemplateExplainer()
        self._init_client()

    @property
    def name(self):
        return self._name if self._client else self._fallback.name

    def _init_client(self):
        try:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not found in environment variables.")
                raise EnvironmentError("GEMINI_API_KEY not set in environment.")
            else:
                logger.info(f"API key found: {api_key[:4]}...")
            self._client = genai.Client(api_key=api_key)
            logger.info(f"Gemini explainer initialised: {self.model_name}")
        except Exception as e:
            logger.exception(f"Could not initialise Gemini client: {e}. Using template fallback.")
            self._client = None

    def _build_prompt(self, text, labels):
        defs = "\n".join(
            f"- {l.replace('_', ' ').title()}: {FALLACY_DEFINITIONS.get(l, 'a logical error')}"
            for l in labels
        )
        return (
            f"The following text has been detected as containing these logical fallacies:\n{defs}\n\n"
            f"Text: \"{text}\"\n\n"
            f"Explain clearly and educationally how this text demonstrates these fallacies. "
            f"Be specific and reference the text directly."
        )

    def explain(self, text, fallacy_labels):
        labels = [l for l in fallacy_labels if l != "no_fallacy"]
        if not labels:
            return "No logical fallacy detected. The reasoning appears valid."
        if self._client is None:
            logger.warning("Gemini client is not initialised. Falling back to template.")
            return self._fallback.explain(text, fallacy_labels)

        prompt = self._build_prompt(text, labels)
        try:
            from google.genai import types
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            )

            if response and response.text:
                return response.text.strip()
            else:
                logger.error("Gemini returned an empty response.")
                return self._fallback.explain(text, fallacy_labels)

        except Exception as e:
            logger.error(f"Gemini API error: {e}. Falling back to template.")
            return self._fallback.explain(text, fallacy_labels)

def build_explainer(config):
    explainer_cfg = config.get("explainer", {})
    provider = explainer_cfg.get("provider", "template")
    if provider == "gemini":
        return GeminiExplainer(
            model_name=explainer_cfg.get("model", "gemini-3.1-pro"),
            max_tokens=explainer_cfg.get("max_output_tokens", 256),
            temperature=explainer_cfg.get("temperature", 0.3),
        )
    else:
        logger.info("Using TemplateExplainer (no API).")
        return TemplateExplainer()
