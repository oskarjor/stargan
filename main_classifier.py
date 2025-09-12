import argparse
import os
from torch.backends import cudnn
import torch
import torch.nn as nn

from classifier import resnet18, resnet34, resnet50, resnet101, resnet152
from data_loader import get_loader


def train(model, celeba_loader_train, celeba_loader_test, config):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    print(f"Dataloader size: {len(celeba_loader_train)}")
    for epoch in range(config.num_epochs):
        total_loss = 0
        for images, labels in celeba_loader_train:
            images = images.to(config.device)
            labels = labels.to(config.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss {total_loss / len(celeba_loader_train)}")

        if epoch % config.save_epoch == 0:
            with torch.no_grad():
                total_loss = 0
                for images, labels in celeba_loader_test:
                    images = images.to(config.device)
                    labels = labels.to(config.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                print(
                    f"Epoch {epoch}, Test Loss {total_loss / len(celeba_loader_test)}"
                )

            print("Saving model...")
            torch.save(model.state_dict(), f"{config.model_save_dir}/model_{epoch}.pth")


def main(config):
    # For fast training.
    # cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    num_classes = len(config.selected_attrs)

    if config.model == "resnet18":
        model = resnet18(num_classes=num_classes)
    elif config.model == "resnet34":
        model = resnet34(num_classes=num_classes)
    elif config.model == "resnet50":
        model = resnet50(num_classes=num_classes)
    elif config.model == "resnet101":
        model = resnet101(num_classes=num_classes)
    elif config.model == "resnet152":
        model = resnet152(num_classes=num_classes)
    else:
        raise ValueError(f"Model {config.model} not found")

    model.to(config.device)

    celeba_loader_train = get_loader(
        config.celeba_image_dir,
        config.attr_path,
        config.selected_attrs,
        config.celeba_crop_size,
        config.image_size,
        config.batch_size,
        "CelebA",
        "train",
        config.num_workers,
    )

    celeba_loader_test = get_loader(
        config.celeba_image_dir,
        config.attr_path,
        config.selected_attrs,
        config.celeba_crop_size,
        config.image_size,
        config.batch_size,
        "CelebA",
        "test",
        config.num_workers,
    )

    train(model, celeba_loader_train, celeba_loader_test, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument("--model", type=str, default="resnet18")

    # Data configuration.
    parser.add_argument("--celeba_image_dir", type=str, default="data/celeba/images")
    parser.add_argument(
        "--attr_path", type=str, default="data/celeba/list_attr_celeba.txt"
    )
    parser.add_argument("--log_dir", type=str, default="classifier/logs")
    parser.add_argument("--model_save_dir", type=str, default="classifier/models")

    # Training configuration.
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--celeba_crop_size", type=int, default=178)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument(
        "--selected_attrs",
        nargs="+",
        type=str,
        default=["Young"],
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_epoch", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    config = parser.parse_args()
    print(config)
    main(config)
