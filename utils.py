import torch
import numpy as np
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def get_metrics(predictions: torch.Tensor, labels: torch.Tensor):
    """Calculate TP, TN, FP, FN between predicted and ground truth binary tensors.

    Args:
        predictions: Binary tensor of model predictions
        labels: Binary tensor of ground truth labels

    Returns:
        tuple: (true_positives, true_negatives, false_positives, false_negatives)
    """
    true_positives = ((predictions == 1) & (labels == 1)).sum().item()
    true_negatives = ((predictions == 0) & (labels == 0)).sum().item()
    false_positives = ((predictions == 1) & (labels == 0)).sum().item()
    false_negatives = ((predictions == 0) & (labels == 1)).sum().item()

    return true_positives, true_negatives, false_positives, false_negatives


def print_metrics(metrics: tuple):
    true_positives, true_negatives, false_positives, false_negatives = metrics
    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(
        f"Accuracy: {(true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)}"
    )


def print_metrics_with_intervention(metrics, attribute_name, intervention_attr):
    print("-" * 63)
    if intervention_attr == attribute_name:
        neg_attribute_name = attribute_name
        attribute_name = "not " + attribute_name
    else:
        neg_attribute_name = "not " + attribute_name
    print(f"Metrics for {attribute_name}")
    print(f"Intervening on {intervention_attr}")
    # Create header row with fixed width columns
    print("-" * 63)
    print(f"{'Real class':<20}| {attribute_name:<20}| {neg_attribute_name:<20}")
    print(f"{'-' * 20:<20}|{'-' * 21}|{'-' * 20}")
    print(
        f"{'Predicted ' + attribute_name:<20}| {metrics['true_positives']:<20}| {metrics['false_positives']:<20}"
    )
    print(
        f"{'Predicted ' + neg_attribute_name:<20}| {metrics['false_negatives']:<20}| {metrics['true_negatives']:<20}"
    )
    print("-" * 63)
    print(
        f"Accuracy: {(metrics['true_positives'] + metrics['true_negatives']) / (metrics['true_positives'] + metrics['true_negatives'] + metrics['false_positives'] + metrics['false_negatives'])}"
    )
    print("-" * 63)


def save_images_with_labels(images, labels, preds, save_dir):
    """Save images in a grid with their corresponding labels on top.

    Args:
        images: Tensor of images
        labels: Tensor of binary labels
        save_dir: Directory to save the images
    """
    # Convert images to grid with more padding
    grid = vutils.make_grid(denorm(images), nrow=4, padding=20, normalize=False)

    # Convert to numpy for matplotlib
    grid = grid.cpu().numpy().transpose((1, 2, 0))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(grid)

    # Calculate positions for labels
    num_cols = 4
    num_rows = len(images) // num_cols + (1 if len(images) % num_cols else 0)
    img_width = images.shape[-1] + 20

    # Add labels above each image
    for idx, label in enumerate(labels):
        row = idx // num_cols
        col = idx % num_cols
        x = col * img_width + img_width / 2
        y = row * img_width + 15
        label_text = "Young" if label.item() == 1 else "Old"
        pred_text = "Young" if preds[idx].item() == 1 else "Old"
        ax.text(
            x,
            y,
            label_text + " -> " + pred_text,
            horizontalalignment="center",
            color="white",
            fontsize=20,
            fontweight="bold",
        )

    # Remove axes
    ax.axis("off")

    # Save figure
    plt.savefig(f"{save_dir}.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def get_metrics_from_saved(preds_path: str):
    preds = np.load(preds_path)
    labels = preds[0]
    preds = preds[1:]
    gen_preds = preds[1:]
    return get_metrics(preds, labels), get_metrics(gen_preds, labels)


if __name__ == "__main__":
    preds_path = "temp/Gray_Hair/preds.npy"
    metrics, gen_metrics = get_metrics_from_saved(preds_path)
    print_metrics(metrics)
    print_metrics(gen_metrics)
