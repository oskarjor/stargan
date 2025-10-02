import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Generator, Discriminator
from classifier import resnet18, resnet50
from data_loader import get_loader
from utils import get_metrics, print_metrics_with_intervention, save_images_with_labels, denorm, generate_all_z_vectors, filter_relevant_z_vectors
from torchvision.utils import make_grid, save_image
import numpy as np
from torchvision import transforms as T
from data_loader import CelebA
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_transform(config):
    transform = []
    transform.append(T.CenterCrop(config["celeba_crop_size"]))
    transform.append(T.Resize(config["image_size"]))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    return transform

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


def calculate_metrics(config, G, classifier, device, intervention_attr, classification_attr, batch_size, split, save_images):
    # LOAD DATALOADER
    celeba_loader = get_loader(
        config["celeba_image_dir"],
        config["attr_path"],
        config["selected_attrs"],
        config["celeba_crop_size"],
        config["image_size"],
        batch_size,
        "CelebA",
        split,
        0,
        classification_attr,
    )


    if intervention_attr is not None:
        if intervention_attr not in config["selected_attrs"]:
            raise ValueError(f"Intervention attribute must be one of {config['selected_attrs']} or None, was {intervention_attr}")
        relevant_idx = config["selected_attrs"].index(intervention_attr)
        related_idx = [
            config["selected_attrs"].index(color)
            for color in ["Gray_Hair", "Black_Hair", "Blond_Hair", "Brown_Hair"]
            if color != intervention_attr and color in config["selected_attrs"]
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

    # TODO: add noising to images to avoid making classification too easy
    with torch.no_grad():
        for i, (images, gen_labels, target) in enumerate(celeba_loader):
            if i > 10 and save_images:
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
            if save_images:
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
            if intervention_attr == classification_attr:
                target = 1 - target

            probs = classifier(images)
            probs = torch.sigmoid(probs)
            preds = torch.round(probs)

            gen_metrics = get_metrics(preds, target)

            total_gen_metrics["true_positives"] += gen_metrics[0]
            total_gen_metrics["true_negatives"] += gen_metrics[1]
            total_gen_metrics["false_positives"] += gen_metrics[2]
            total_gen_metrics["false_negatives"] += gen_metrics[3]

            if save_images:
                save_images_with_labels(
                    images,
                    target,
                    preds,
                    f"temp/{intervention_attr}/{i}_new_images",
                )

        print_metrics_with_intervention(total_metrics, classification_attr, None)
        print("\n\n")
        print_metrics_with_intervention(total_gen_metrics, classification_attr, intervention_attr)

def single_image_all_variations(image_idx, config, G, classifier, image_save_path, device, classification_attr, D, celeba_dataset, z_vectors, split, max_batch_size):

    if classification_attr in config["selected_attrs"]:
        classification_attr_idx = config["selected_attrs"].index(classification_attr)
    else:
        classification_attr_idx = None

    image, label, target = celeba_dataset[image_idx]

    os.makedirs(image_save_path, exist_ok=True)
    label_str = '_'.join(map(str, label.int().tolist()))
    
    with torch.no_grad():
        logits = classifier(image.unsqueeze(0).to(device))
        probs = torch.sigmoid(logits)
        out_src, _ = D(image.unsqueeze(0).to(device))
        real_src = out_src.mean()
        save_image(denorm(image), os.path.join(image_save_path, f"real_z_{label_str}_prob_{probs.item():.3f}_target_{target.item():.0f}_real_{real_src.item():.3f}.png"))

    n_images = z_vectors.shape[0]
    # Pre-allocate on CPU (lighter memory footprint)
    images = image.repeat(max_batch_size, 1, 1, 1)
    targets = target.repeat(max_batch_size, 1)

    del image, label

    dataloader = DataLoader(z_vectors, batch_size=max_batch_size, pin_memory=True)

    loss = 0
    realness = 0
    
    # Pre-compute target value (constant across all batches)
    target_val = target.item()

    with torch.no_grad():
        losses = torch.zeros(n_images)
        realnesses = torch.zeros(n_images)
        for i, batch_z_vectors in enumerate(dataloader):
            batch_size = batch_z_vectors.shape[0]
            current_images = images[:batch_size].to(device, non_blocking=True)
            current_targets = targets[:batch_size].to(device, non_blocking=True)
            batch_z_vectors = batch_z_vectors.to(device, non_blocking=True)


            # Generate images
            new_images = G(current_images, batch_z_vectors)
            
            # Compute metrics in batch
            logits = classifier(new_images)
            out_src, _ = D(new_images)

            if classification_attr_idx is not None:
                current_targets = batch_z_vectors[:, classification_attr_idx].unsqueeze(1).to(torch.float32)

            
            # Accumulate metrics (minimize .item() calls)
            _losses = F.binary_cross_entropy_with_logits(logits, current_targets, reduction="none").mean(dim=tuple(range(1, logits.dim())))
            loss += _losses.mean().item() * batch_size
            realness += out_src.mean().item() * batch_size
            
            # Batch denormalize BEFORE moving to CPU
            new_images_denorm = denorm(new_images)
            
            # Single transfer to CPU for all tensors
            probs = torch.sigmoid(logits).cpu()
            batch_z_vectors_cpu = batch_z_vectors.cpu()
            new_images_cpu = new_images_denorm.cpu()
            
            # Get real_src for each image in the batch
            out_src_dims = out_src.dim()
            real_src_val = out_src.mean(dim=tuple(range(1, out_src_dims)))

            losses[i*max_batch_size:i*max_batch_size + batch_size] = _losses.detach().cpu()
            realnesses[i*max_batch_size:i*max_batch_size + batch_size] = real_src_val.detach().cpu()

            target_vals = current_targets.detach().cpu()
            # Clear GPU memory immediately
            del current_images, current_targets, batch_z_vectors, new_images, logits, out_src, new_images_denorm

            # Save images (this is still I/O bound, but optimized)
            for i in range(batch_size):
                # Use list comprehension for faster string conversion
                z_str = '_'.join(str(int(x)) for x in batch_z_vectors_cpu[i].tolist())
                filename = f"z_{z_str}_prob_{probs[i].item():.3f}_target_{target_vals[i].item():.0f}_real_{real_src_val[i].item():.3f}.png"
                save_path = os.path.join(image_save_path, filename)
                save_image(new_images_cpu[i], save_path)
            
            del new_images_cpu, probs, batch_z_vectors_cpu

    # Compute means
    mean_loss = loss / n_images
    mean_realness = realness / n_images
    
    with open(f"{image_save_path}/losses.npy", "wb") as f:
        np.save(f, losses)
    with open(f"{image_save_path}/realnesses.npy", "wb") as f:
        np.save(f, realnesses)
    return mean_loss, mean_realness

def all_images_all_variations(config, image_save_dir, device, classification_attr, G, D, classifier, split="test", max_batch_size=16, num_images=100):
    # Save config attributes to json for reference
    config_dict = {
        "attributes": config["selected_attrs"],
        "classification_attr": classification_attr,
        "num_images": num_images,
    }

    os.makedirs(image_save_dir, exist_ok=True)
        
    config_path = os.path.join(image_save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

        transform = get_transform(config)

    celeba_dataset = CelebA(config["celeba_image_dir"], config["attr_path"], config["selected_attrs"], transform, split, classification_attr)
    z_vectors = generate_all_z_vectors(config, filter_hair=(classification_attr == "Young")).to(device)


    for image_idx in tqdm(range(num_images)):
        image_save_path = f"{image_save_dir}/{image_idx}"
        filtered_z_vectors = filter_relevant_z_vectors(config, z_vectors, initial_z_vector=celeba_dataset[image_idx][1])
        if filtered_z_vectors is None:
            continue
        filtered_z_vectors = filtered_z_vectors.to(device)
        single_image_all_variations(image_idx, config, G, classifier, image_save_path, device, classification_attr, D, celeba_dataset, filtered_z_vectors, split, max_batch_size)

if __name__ == "__main__":
    # HYPERPARAMETERS
    classifier_model = "young_noise_00"
    classifier_epoch = "model_10"
    generator_model = "bal_gray"
    generator_epoch = "200000"
    max_batch_size = 32
    classification_attr = "Young"
    num_images = 1000
    
    device = "mps"
    classifier_model_type = "resnet50"
    split = "test"
    intervention_attr = None
    save_images = True

    # INFERRED HYPERPARAMETERS (DO NOT CHANGE)
    image_save_dir = f"generated_images/{generator_model}/{classifier_model}/{classifier_epoch}_only_relevant_z_vectors"
    generator_model_dir = f"stargan_celeba_256/models/{generator_model}"
    generator_model_name = f"{generator_epoch}"
    classifier_model_dir = f"classifier/models/{classifier_model}"
    classifier_model_name = f"{classifier_epoch}.pth"

    # LOAD MODELS AND CONFIG
    config, G, D = load_config_and_model(
        model_dir=generator_model_dir, model_name=generator_model_name
    )
    G = G.to(device)
    D = D.to(device).eval()

    # LOAD CLASSIFIER
    classifier = get_resnet_classifier(os.path.join(classifier_model_dir, classifier_model_name), classifier_model_type).to(
        device  
    ).eval()

    all_images_all_variations(config, image_save_dir, device, classification_attr, G, D, classifier, split, max_batch_size, num_images)

    # calculate_metrics(config, G, classifier, device, intervention_attr, classification_attr, batch_size, split, save_images)
