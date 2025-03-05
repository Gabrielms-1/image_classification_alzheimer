import torch
import torch.nn as nn
from dataset import FolderBasedDataset
from torch.utils.data import DataLoader
from network import AlexNet
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = AlexNet(num_classes=4)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model

def main(model_path, images_path):
    model = load_model(model_path)

    dataset = FolderBasedDataset(images_path, 224)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=False)

    all_predictions = []
    all_labels = []

    correct = 0
    for images, labels, _ in dataloader:
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())
        correct += (predictions == labels).sum().item()
    classes = dataset.int_to_label_map
    confusion_matrix = torch.zeros(len(classes), len(classes), dtype=torch.int32)

    accuracy = correct / len(all_predictions)

    for idx, pred_idx in enumerate(all_predictions):
        if pred_idx != all_labels[idx]:
            print(f"ERROR: Image {idx} - Predicted: {classes[pred_idx]} - True: {classes[all_labels[idx]]}")
        else:
            print(f"Correct: Image {idx} - Predicted: {classes[pred_idx]} - True: {classes[all_labels[idx]]}")
        confusion_matrix[all_labels[idx], pred_idx] += 1

    class_names = [classes[i] for i in range(len(classes))]
    df_cm = pd.DataFrame(confusion_matrix.numpy(), index=class_names, columns=class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
    
    print(f"Accuracy: {accuracy}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/best_model.pth")
    parser.add_argument("--images_path", type=str, default="data/raw/test")
    args = parser.parse_args()

    main(args.model_path, args.images_path)