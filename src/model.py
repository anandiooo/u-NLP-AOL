import logging

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

# defines the neural network architecture
class FallacyClassifier(nn.Module):
    # initialises the class instance
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

    # initialises linear layer weights
    def _init_weights(self):
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # performs forward pass on input
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_repr)

# loads model weights from checkpoint
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
