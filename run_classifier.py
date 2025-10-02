from full_pipeline import get_resnet_classifier, load_config_and_model, get_transform, get_metrics
import os
from PIL import Image
from decode_all_variations import file_to_repr
import torch
from utils import print_metrics, denorm
import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms as T

def run_classifier(classifiers, image_dir, config, classification_attr, device):
    transform = []
    transform.append(T.Resize(config["image_size"]))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    subfolders = [f.path for f in os.scandir(image_dir) if f.is_dir()]
    subfolders.sort(key=lambda x: int(os.path.basename(x)))
    all_metrics = []
    for classifier in tqdm.tqdm(classifiers):
        total_metrics = {
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0
        }
        for subfolder in subfolders:
            images = [f.path for f in os.scandir(subfolder) if f.is_file()]
            for image_path in images:
                file_name = os.path.basename(image_path)
                if file_name.endswith(".png") and file_name.startswith("z_"):
                    z_vector, prob, target, realness = file_to_repr(file_name, len(config["selected_attrs"]))
                    target = torch.tensor([int(target)])
                    image = Image.open(image_path)
                    image = transform(image)
                    image = image.unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs = classifier(image)
                        preds = torch.round(torch.sigmoid(probs)).cpu()
                    metrics = get_metrics(preds, target)
                    total_metrics["true_positives"] += metrics[0]
                    total_metrics["true_negatives"] += metrics[1]
                    total_metrics["false_positives"] += metrics[2]
                    total_metrics["false_negatives"] += metrics[3]
        all_metrics.append(total_metrics)
    return all_metrics


if __name__ == "__main__":
        # HYPERPARAMETERS
    classifier_models = ["young_noise_00", "young_noise_01", "young_noise_05"]
    classifier_epochs = ["model_0", "model_5", "model_10"]
    max_batch_size = 32
    classification_attr = "Young"
    image_dir = "cherry_picked"
    
    device = "mps"
    classifier_model_type = "resnet50"
    split = "test"
    save_images = True

    classifiers = []
    classifier_names = []

    for classifier_model in classifier_models:
        for epoch in classifier_epochs:
            classifier_name = f"{classifier_model}_{epoch}"
            classifier_names.append(classifier_name)
            # INFERRED HYPERPARAMETERS (DO NOT CHANGE)
            classifier_model_dir = f"classifier/models/{classifier_model}"
            classifier_model_name = f"{epoch}.pth"

            # LOAD CLASSIFIER
            classifier = get_resnet_classifier(os.path.join(classifier_model_dir, classifier_model_name), classifier_model_type).to(
                device  
            ).eval()
            classifiers.append(classifier)
    
    config, _, _ = load_config_and_model(
        model_dir="stargan_celeba_256/models/bal_gray", model_name="200000"
    )

    all_metrics = run_classifier(classifiers, image_dir, config, classification_attr, device)
    for classifier_name, metrics in zip(classifier_names, all_metrics):
        print(f"{classifier_name}:")
        print(metrics)
        print("Accuracy: ", (metrics["true_positives"] + metrics["true_negatives"]) / (metrics["true_positives"] + metrics["true_negatives"] + metrics["false_positives"] + metrics["false_negatives"]))
        print()
        