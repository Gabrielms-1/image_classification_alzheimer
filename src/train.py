import torch
import torch.optim as optim
import os
import argparse
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
import boto3

from dataset import FolderBasedDataset, create_data_loaders
from network import AlexNet
from config import Config

def create_dataloaders(train_dir, val_dir, resize, batch_size):
    train_dataset = FolderBasedDataset(train_dir, resize)
    val_dataset = FolderBasedDataset(val_dir, resize)

    train_dataloader, val_dataloader = create_data_loaders(train_dataset, val_dataset, batch_size)

    return train_dataloader, val_dataloader

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io

def plot_metrics(train_losses, train_accs, val_losses, val_accs, args):
    epochs = range(1, int(args.epochs) + 1)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Loss Curve during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-o', label='Validation Accuracy')
    plt.title('Accuracy Curve during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, "training_metrics.png"))
    plt.show()
    
    print(f"Metrics plot saved as {os.path.join(args.checkpoint_dir, 'training_metrics.png')}")  

    # Convert the plot to a PIL image
    canvas = FigureCanvas(plt.gcf())
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    image = Image.open(buf)

    return image

def evaluate_model(model, val_dataloader, criterion, device):
    model.eval()

    val_loss = 0
    correct_predictions = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.detach(), 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        
        average_loss = val_loss / len(val_dataloader.dataset)
        accuracy = correct_predictions / total

    return average_loss, accuracy


def train_model(model, epochs, train_dataloader, val_dataloader, criterion, optimizer, device):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

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

        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"-" * 50)
        print(f"EPOCH: {i+1}")
        print(f"- train_loss: {epoch_loss} | train_accuracy: {correct_predictions / total_examples}")
        print(f"- val_loss: {val_loss} | val_accuracy: {val_acc}" )
        print(f"-" * 50)

        epoch_accuracy = correct_predictions / total_examples
        train_accuracies.append(epoch_accuracy)
    
        if (i + 1) % 10 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": i+1,
                    "loss": epoch_loss,
                    "accuracy": epoch_accuracy,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "wandb_config": wandb.config
                },
                os.path.join(args.checkpoint_dir,  f"checkpoint_{i+1}.pth")
            )
            print(f"Checkpoint {i+1} saved")
        wandb.log({
            "epoch": i+1,
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    final_model_path = os.path.join(args.model_dir, "final_alexnet_model.pth")
    torch.save(model.state_dict(), final_model_path)
    
    return train_losses, train_accuracies, val_losses, val_accuracies


def main(args):    
    wandb.init(
        project=f"{args.model_name}",
        name=f"{args.model_name}_{datetime.now().strftime('%d-%m-%Y_%H-%M')}",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_classes": args.num_classes,
            "resize": args.resize,
            "optimizer": Config.OPTIMIZER,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_dataloader, val_dataloader = create_dataloaders(args.train, args.val, args.resize, args.batch_size)

    model = AlexNet(num_classes=args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    
    if Config.OPTIMIZER == "Adam":
        optimizer = optim.Adam(model.parameters(), args.learning_rate, weight_decay=Config.WEIGHT_DECAY)

    if Config.OPTIMIZER == "SGD":
        optimizer = optim.SGD(model.parameters(), args.learning_rate, momentum=Config.SGD_MOMENTUM, weight_decay=Config.WEIGHT_DECAY)

    train_losses, train_accuracies, val_losses, val_accuracies = train_model(model, args.epochs, train_dataloader, val_dataloader, criterion, optimizer, device)

    print(f" * Training completed. Saving metrics plot...")

    metrics_plot = plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, args)

    wandb.log({
        "metrics_plot": wandb.Image(metrics_plot)
    })

    wandb.finish()

    s3 = boto3.client('s3')
    local_wandb_dir = "/opt/ml/code/wandb"
    s3_bucket = "cad-brbh-datascience"
    s3_dest_prefix = "alzheimer_images/checkpoints/wandb"
    for root, _, files in os.walk(local_wandb_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_wandb_dir)
            s3_key = os.path.join(s3_dest_prefix, relative_path)
            s3.upload_file(local_path, s3_bucket, s3_key)

    # local_model_dir = "/tmp/wandb"
    # s3_bucket = "cad-brbh-datascience"
    # s3_dest_prefix = "alzheimer_images/checkpoints/wandb"
    # for root, _, files in os.walk(local_model_dir):
    #     for file in files:
    #         local_path = os.path.join(root, file)
    #         relative_path = os.path.relpath(local_path, local_wandb_dir)
    #         s3_key = os.path.join(s3_dest_prefix, relative_path)
    #         s3.upload_file(local_path, s3_bucket, s3_key)


    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default=Config.PROJECT_NAME)
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=Config.LEARNING_RATE)
    parser.add_argument("--model_dir", default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val", default=os.environ["SM_CHANNEL_VAL"])
    parser.add_argument("--num_classes", type=int, default=Config.NUM_CLASSES)
    parser.add_argument("--resize", type=int, default=Config.RESIZE)
    parser.add_argument("--checkpoint_dir", default="/opt/ml/checkpoints")
    args = parser.parse_args()

    main(args)