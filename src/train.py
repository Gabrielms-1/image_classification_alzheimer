import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score, f1_score
import os
import argparse
import wandb

from dataset import FolderBasedDataset, create_data_loaders
from network import AlexNet

def create_dataloaders(train_dir, val_dir, resize, batch_size):
    train_dataset = FolderBasedDataset(train_dir, resize)
    val_dataset = FolderBasedDataset(val_dir, resize)

    train_dataloader, val_dataloader = create_data_loaders(train_dataset, val_dataset, batch_size)

    return train_dataloader, val_dataloader

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device):
    
    model.train()

    for i in range(args.epochs):

        for images, labels, _ in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    
    pass

def main(args):
    # wandb.init(
    #     project=f"",
    #     name=f"",
    #     config={
    #         "epochs": args.epochs,
    #         "batch_size": args.batch_size,
    #         "learning_rate": args.learning_rate,
    #         "num_classes": args.num_classes,
    #         "resize": args.resize
    #     }
    # )

    os.makedirs("/opt/ml/checkpoints", exist_ok=True)

    train_dataloader, val_dataloader = create_dataloaders(args.train, args.val, args.resize, args.batch_size)

    model = AlexNet(num_classes=args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate, weight_decay=0.0001)

    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device)

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