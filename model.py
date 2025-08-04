import torch
import torch.nn as nn
from transformers import AutoModel


class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=12, pretrained=True):
        super(CNN, self).__init__()
        # Load RadDINO model as backbone
        self.model = AutoModel.from_pretrained(
            "microsoft/rad-dino",
            attn_implementation="sdpa",
            num_labels=n_outputs,
        )

        # Optional: print hidden size for debugging
        print(f"RadDINO hidden size: {self.model.config.hidden_size}")

        # Classification head
        self.classifier = nn.Linear(self.model.config.hidden_size, n_outputs)

    def forward(self, x):
        # Forward through RadDINO backbone
        outputs = self.model(x)

        # Extract relevant features
        if hasattr(outputs, "logits"):
            feats = outputs.logits
        elif hasattr(outputs, "pooler_output"):
            feats = outputs.pooler_output
        else:
            # fallback to first hidden state mean
            feats = outputs.last_hidden_state.mean(dim=1)

        # Classifier
        return self.classifier(feats)
