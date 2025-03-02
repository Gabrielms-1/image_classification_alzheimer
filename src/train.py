import torch
import torch.optim as optim
import os
import argparse
import wandb
from datetime import datetime

from dataset import FolderBasedDataset, create_data_loaders
from network import AlexNet

def create_dataloaders(train_dir, val_dir, resize, batch_size):
    train_dataset = FolderBasedDataset(train_dir, resize)
    val_dataset = FolderBasedDataset(val_dir, resize)

    train_dataloader, val_dataloader = create_data_loaders(train_dataset, val_dataset, batch_size)

    return train_dataloader, val_dataloader

def train_model(model, epochs, train_dataloader, val_dataloader, criterion, optimizer, device):
    train_losses = []
    train_accuracies = []
    model.train()

    for i in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_examples = 0

        for images, labels, _ in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.detach(), 1) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()   

            epoch_loss += loss.item() * images.size(0) # loss x batch size

            correct_predictions += (predicted == labels).sum().item()
            total_examples += labels.size(0)

        epoch_loss = epoch_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_loss)

        epoch_accuracy = correct_predictions / total_examples
        train_accuracies.append(epoch_accuracy)
    
        wandb.log({
            "epoch": i+1,
            "loss": epoch_loss,
            "accuracy": epoch_accuracy
        })
    return train_losses, train_accuracies


def main(args):
    wandb.init(
        project=f"{args.model_name}",
        name=f"{args.model_name}_{datetime.now().strftime('%d-%m-%Y_%H-%M')}",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_classes": args.num_classes,
            "resize": args.resize
        }
    )

    os.makedirs("/opt/ml/checkpoints", exist_ok=True)

    train_dataloader, val_dataloader = create_dataloaders(args.train, args.val, args.resize, args.batch_size)

    model = AlexNet(num_classes=args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate, weight_decay=0.0001)

    train_losses = train_model(model, args.epochs, train_dataloader, val_dataloader, criterion, optimizer, device)

    wandb.finish()

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    #parser.add_argument("--task")
    parser.add_argument("--model_dir")
    parser.add_argument("--train")
    parser.add_argument("--val")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--resize", type=int)

    args = parser.parse_args()

    main(args)