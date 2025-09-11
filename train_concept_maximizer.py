import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Generator, Discriminator


def load_config_and_model(model_dir=None, model_name=None):
    """Load config and model from a directory.

    Args:
        model_dir (str): Directory containing config.json and model checkpoints.
                        If None, returns None, None

    Returns:
        config (dict): Configuration dictionary
        model (torch.nn.Module): Loaded model if model_dir provided, else None
    """
    if model_dir is None:
        return None, None

    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    G = Generator(config["g_conv_dim"], config["c_dim"], config["g_repeat_num"])
    D = Discriminator(
        config["image_size"],
        config["d_conv_dim"],
        config["c_dim"],
        config["d_repeat_num"],
    )

    G_path = os.path.join(model_dir, "{}-G.ckpt".format(model_name))
    D_path = os.path.join(model_dir, "{}-D.ckpt".format(model_name))
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    return config, G, D


class DummyClassifier(nn.Module):
    def __init__(self, config):
        super(DummyClassifier, self).__init__()
        self.config = config
        self.fc = nn.Linear(config["image_size"] * config["image_size"] * 3, 1)

    def forward(self, x):
        return F.sigmoid(self.fc(x.view(x.size(0), -1)))


class ConceptMaximizer(nn.Module):
    def __init__(self, config, G, classifier, batch_size=1, low=-1, high=1):
        super(ConceptMaximizer, self).__init__()
        self.config = config
        self.classifier = classifier
        self.batch_size = batch_size
        self.low = low
        self.high = high

        self.concepts = nn.Parameter(
            torch.randn(self.batch_size, config["c_dim"], requires_grad=True)
        )
        self.G = G.eval()
        self.freeze_generator_and_discriminator()
        self.freeze_classifier()

    def freeze_generator_and_discriminator(self):
        for param in self.G.parameters():
            param.requires_grad = False

    def freeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = False

    def constrain_concepts(self):
        return self.low + (self.high - self.low) * F.sigmoid(self.concepts)

    def forward(self, x):
        concepts = self.constrain_concepts()
        return self.classifier(self.G(x, concepts))


if __name__ == "__main__":
    # Load models and config
    config, G, _ = load_config_and_model(
        model_dir="stargan_celeba/models", model_name="100"
    )

    # Create dummy data
    batch_size = 1
    x = torch.randn(batch_size, 3, config["image_size"], config["image_size"])
    concepts = torch.randn(batch_size, config["c_dim"], requires_grad=True)

    # Create dummy classifier and concept maximizer
    classifier = DummyClassifier(config)
    concept_maximizer = ConceptMaximizer(config, G, classifier, batch_size)

    # Train concept maximizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, concept_maximizer.parameters()), lr=0.1
    )
    criterion = nn.BCELoss()
    for iter_num in range(1000):
        optimizer.zero_grad()
        cls_probs = concept_maximizer(x)
        loss = criterion(cls_probs, torch.zeros_like(cls_probs))
        loss.backward()
        optimizer.step()
        if iter_num % 100 == 0:
            print(f"Iteration {iter_num}: Loss {loss.item()}")
            print(f"Concepts: {concept_maximizer.constrain_concepts().data}")
            print(f"Cls Probs: {cls_probs.data}")
