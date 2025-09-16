from torch.utils import data
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(
        self, image_dir, attr_path, selected_attrs, transform, mode, target_class=None
    ):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.target_class = target_class
        self.preprocess()

        if mode == "train":
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, "r")]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            target = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == "1")

            if self.target_class is not None:
                idx = self.attr2idx[self.target_class]
                target.append(values[idx] == "1")

            if (i + 1) < 2000:
                if self.target_class is not None:
                    self.test_dataset.append([filename, label, target])
                else:
                    self.test_dataset.append([filename, label])
            else:
                if self.target_class is not None:
                    self.train_dataset.append([filename, label, target])
                else:
                    self.train_dataset.append([filename, label])

        print("Finished preprocessing the CelebA dataset...")

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset
        if self.target_class is not None:
            filename, label, target = dataset[index]
        else:
            filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        if self.target_class is not None:
            return (
                self.transform(image),
                torch.FloatTensor(label),
                torch.FloatTensor(target),
            )
        else:
            return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(
    image_dir,
    attr_path,
    selected_attrs,
    crop_size=178,
    image_size=128,
    batch_size=16,
    dataset="CelebA",
    mode="train",
    num_workers=1,
    target_class=None,
    balance_attr=None,
):
    """Build and return a data loader."""
    transform = []
    if mode == "train":
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == "CelebA":
        dataset = CelebA(
            image_dir, attr_path, selected_attrs, transform, mode, target_class
        )
    elif dataset == "RaFD":
        dataset = ImageFolder(image_dir, transform)

    sampler = None
    if dataset.__class__.__name__ == "CelebA" and balance_attr is not None:
        if balance_attr not in selected_attrs:
            raise ValueError(
                f"balance_attr '{balance_attr}' must be included in selected_attrs to balance by it."
            )
        attr_idx_in_selected = selected_attrs.index(balance_attr)
        records = dataset.train_dataset if mode == "train" else dataset.test_dataset
        # Collect binary labels for the balancing attribute
        labels = []
        for item in records:
            label_list = item[1]  # item structure: [filename, label, (optional) target]
            label_val = 1 if bool(label_list[attr_idx_in_selected]) else 0
            labels.append(label_val)
        # Compute class counts and inverse-frequency weights
        num_pos = sum(labels)
        num_neg = len(labels) - num_pos
        # Avoid division by zero
        if num_pos == 0 or num_neg == 0:
            # Degenerate case: cannot balance a single-class subset
            # Fall back to regular shuffling
            sampler = None
        else:
            weight_for_class = {0: 1.0 / num_neg, 1: 1.0 / num_pos}
            sample_weights = [weight_for_class[y] for y in labels]
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(labels), replacement=True
            )
            print(
                f"Balanced sampler created with {num_pos} positive and {num_neg} negative samples."
            )

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and mode == "train"),
        sampler=sampler,
        num_workers=num_workers,
    )
    return data_loader
