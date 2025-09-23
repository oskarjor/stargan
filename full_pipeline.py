import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Generator, Discriminator
from classifier import resnet18, resnet50
from data_loader import get_loader
from utils import get_metrics, print_metrics_with_intervention, save_images_with_labels
from torchvision.utils import make_grid, save_image
import numpy as np


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


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


def get_resnet_classifier(path=None, resnet_type="resnet18"):
    if path is None:
        raise ValueError("Path to resnet classifier is required")
    if resnet_type == "resnet18":
        resnet = resnet18(num_classes=1)
    elif resnet_type == "resnet50":
        resnet = resnet50(num_classes=1)
    else:
        raise ValueError(f"Resnet type {resnet_type} not found")

    resnet.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    return resnet


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


def swap_hair_color(labels, relevant_idx, related_idx):
    new_labels = labels.clone()
    # set relevant hair color to opposite
    new_labels[:, relevant_idx] = 1 - new_labels[:, relevant_idx]
    # where we turned off, turn on a random related hair color
    random_related_idx = related_idx[0][torch.randint(0, len(related_idx[0]), (1,))]
    no_hair_color_indices = torch.where(new_labels[:, relevant_idx] == 0)[0]
    new_labels[no_hair_color_indices, random_related_idx] = 1
    return new_labels


def save_images_and_preds(images, preds, new_images, new_preds, target, save_dir):
    # save images
    os.makedirs(save_dir, exist_ok=True)
    save_image(make_grid(denorm(new_images)), f"{save_dir}/new_images.png")
    save_image(make_grid(denorm(images)), f"{save_dir}/original_images.png")

    all_preds = torch.cat([target, preds, new_preds], dim=0)
    # save preds to txt file
    np.save(f"{save_dir}/preds.npy", all_preds.detach().cpu().numpy())


if __name__ == "__main__":
    # Load models and config
    config, G, _ = load_config_and_model(
        model_dir="stargan_celeba_256/models/no_young_bal_gray", model_name="200000"
    )

    device = "mps"
    G = G.to(device)

    classifier = get_resnet_classifier("classifier/models/res50_epoch10_256.pth", "resnet50").to(
        device  
    )
    split = "test"
    celeba_loader = get_loader(
        config["celeba_image_dir"],
        config["attr_path"],
        config["selected_attrs"],
        config["celeba_crop_size"],
        config["image_size"],
        16,
        "CelebA",
        split,
        config["num_workers"],
        "Young",
    )

    print(config["selected_attrs"])
    intervention_attr = "Gray_Hair"

    if intervention_attr is not None:
        relevant_idx = config["selected_attrs"].index(intervention_attr)
        related_idx = [
            config["selected_attrs"].index(color)
            for color in ["Gray_Hair", "Black_Hair", "Blond_Hair", "Brown_Hair"]
            if color != intervention_attr
        ]

        relevant_idx = torch.tensor([relevant_idx]).to(device)
        related_idx = torch.tensor([related_idx]).to(device)

    total_metrics = {
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }
    total_gen_metrics = {
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }

    is_saving = True

    # TODO: add noising to images to avoid making classification too easy

    for i, (images, gen_labels, target) in enumerate(celeba_loader):
        if is_saving:
            print(f"Processing batch {i}")
            if i > 10:
                break
        images = images.to(device)
        gen_labels = gen_labels.to(device)
        target = target.to(device)
        probs = classifier(images)
        probs = torch.sigmoid(probs)
        preds = torch.round(probs)
        metrics = get_metrics(preds, target)

        total_metrics["true_positives"] += metrics[0]
        total_metrics["true_negatives"] += metrics[1]
        total_metrics["false_positives"] += metrics[2]
        total_metrics["false_negatives"] += metrics[3]
        if is_saving:
            os.makedirs(f"temp/{intervention_attr}", exist_ok=True)
            save_images_with_labels(
                images, target, preds, f"temp/{intervention_attr}/{i}_original_images"
            )

        if intervention_attr in ["Gray_Hair", "Black_Hair", "Blond_Hair", "Brown_Hair"]:
            gen_labels = swap_hair_color(gen_labels, relevant_idx, related_idx)
        elif intervention_attr == None:
            pass
        else:
            gen_labels[:, relevant_idx] = 1 - gen_labels[:, relevant_idx]

        images = G(images, gen_labels)
        if intervention_attr == "Young":
            target = 1 - target

        probs = classifier(images)
        probs = torch.sigmoid(probs)
        preds = torch.round(probs)

        gen_metrics = get_metrics(preds, target)

        total_gen_metrics["true_positives"] += gen_metrics[0]
        total_gen_metrics["true_negatives"] += gen_metrics[1]
        total_gen_metrics["false_positives"] += gen_metrics[2]
        total_gen_metrics["false_negatives"] += gen_metrics[3]

        if is_saving:
            save_images_with_labels(
                images,
                target,
                preds,
                f"temp/{intervention_attr}/{i}_new_images",
            )

    print_metrics_with_intervention(total_metrics, "Young", None)
    print("\n\n")
    print_metrics_with_intervention(total_gen_metrics, "Young", intervention_attr)
