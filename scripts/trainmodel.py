import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import copy
import matplotlib.pyplot as plt
from utilities import fetch_csv_data

class ShapeDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.targets[index]
        if self.transform:
            img = self.transform(img)
        return img, label

def load_data_from_csv(csv_list, image_size=(64, 64)):
    images = []
    targets = []
    label_dict = {'circle': 0, 'ellipse': 1, 'rectangle': 2, 'polygon': 3, 'star': 4}

    for file in csv_list:
        shapes = fetch_csv_data(file)
        for shape in shapes:
            fig, ax = plt.subplots(figsize=(image_size[0] / 100, image_size[1] / 100), dpi=100)
            ax.set_xlim(0, image_size[0])
            ax.set_ylim(0, image_size[1])

            for coordinates in shape:
                if coordinates.shape[1] < 2:
                    continue
                ax.plot(coordinates[:, 0], coordinates[:, 1], 'k', linewidth=2)

            ax.axis('off')
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(image_size[1], image_size[0], 3)
            images.append(Image.fromarray(img_array))
            shape_name = os.path.basename(file).split('.')[0]
            targets.append(label_dict[shape_name])
            plt.close(fig)

    return images, targets

def train_network(model, loaders, loss_func, optimizer, device, num_epochs=10, early_stop_patience=3):
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            total_loss = 0.0
            correct_predictions = 0

            for imgs, labels in loaders[phase]:
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs)
                    _, predictions = torch.max(outputs, 1)
                    loss = loss_func(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                correct_predictions += torch.sum(predictions == labels.data)

            avg_loss = total_loss / len(loaders[phase].dataset)
            avg_accuracy = correct_predictions.double() / len(loaders[phase].dataset)

            print(f'{phase} Loss: {avg_loss:.4f} Accuracy: {avg_accuracy:.4f}')

            if phase == 'val':
                if avg_accuracy > best_accuracy:
                    best_accuracy = avg_accuracy
                    best_weights = copy.deepcopy(model.state_dict())
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

        if no_improve_epochs >= early_stop_patience:
            print("Early stopping triggered")
            break

    print(f'Best Validation Accuracy: {best_accuracy:.4f}')
    model.load_state_dict(best_weights)
    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    csv_files = [
        '/content/drive/MyDrive/curve_project/data/circle.csv',
        '/content/drive/MyDrive/curve_project/data/ellipse.csv',
        '/content/drive/MyDrive/curve_project/data/rectangle.csv',
        '/content/drive/MyDrive/curve_project/data/polygon.csv',
        '/content/drive/MyDrive/curve_project/data/star.csv'
    ]

    images, targets = load_data_from_csv(csv_files, image_size=(64, 64))

    train_images, val_images, train_targets, val_targets = train_test_split(
        images, targets, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = ShapeDataset(train_images, train_targets, transform=transform)
    val_dataset = ShapeDataset(val_images, val_targets, transform=transform)

    data_loaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    }

    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)  # 5 shape categories

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = train_network(model, data_loaders, criterion, optimizer, device, num_epochs=10, early_stop_patience=3)

    model_save_path = '/content/drive/MyDrive/curve_project/models/shape_recognition_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    main()
