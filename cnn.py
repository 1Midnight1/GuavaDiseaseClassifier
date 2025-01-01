import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import ToDtype
from torchvision import transforms
from torchvision.io import image

torch.manual_seed(42)

DATASET_DIR = "./guava_dataset/GuavaDiseaseDataset/GuavaDiseaseDataset"

CLASSES = {"Anthracnose": 0, "fruit_fly": 1, "healthy_guava": 2}
CLASSES_NAMES = {0: "Anthracnose", 1: "Fruit Fly", 2: "Healthy Guava"}


def load_guava_dataset(balanced: bool = False) -> tuple[DataLoader, DataLoader]:
    test_dir = DATASET_DIR + "/test"
    train_dir = DATASET_DIR + "/train"
    validation_dir = DATASET_DIR + "/val"

    transform = transforms.Compose([ToDtype(dtype=torch.float32), transforms.Normalize((0.5,), (0.5,))])

    training_dataset = ImageDataset([train_dir, validation_dir], classes=CLASSES, transform=transform, balanced=balanced)
    test_dataset = ImageDataset([test_dir], classes=CLASSES, transform=transform)

    training_dataloader = DataLoader(dataset=training_dataset, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=10)

    return training_dataloader, test_dataloader


class ImageDataset(Dataset):
    def __init__(self, directories: list[str], classes: dict[str, int], transform: transforms.Compose, balanced: bool = False):
        self.directories = directories
        self.classes = classes
        self.images = []
        self.labels = []
        if balanced:
            self.load_image_dataset_balanced(transform)
        else:
            self.load_image_dataset(transform)

    def load_image_dataset(self, transform: transforms.Compose):
        for cls in self.classes.keys():
            for dir in self.directories:
                for img_path in os.listdir(f"{dir}/{cls}"):
                    self.images.append(transform(image.decode_image(f"{dir}/{cls}/{img_path}")))
                    self.labels.append(self.classes[cls])

    def load_image_dataset_balanced(self, transform: transforms.Compose):
        class_counts = {}
        for cls in self.classes.keys():
            class_counts[cls] = 0
            for dir in self.directories:
                class_counts[cls] += len(os.listdir(f"{dir}/{cls}"))

        min_count = min(class_counts.values())

        for cls in self.classes.keys():
            for dir in self.directories:
                imgs = list(os.listdir(f"{dir}/{cls}"))
                target = round(len(imgs) * min_count / class_counts[cls])
                skip = len(imgs) // target
                taken_from_dir = 0
                for img_path in imgs[::skip]:
                    if taken_from_dir >= target:
                        break
                    self.images.append(transform(image.decode_image(f"{dir}/{cls}/{img_path}")))
                    self.labels.append(self.classes[cls])
                    taken_from_dir += 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class GuavaDiseaseClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        ACTIVATION_FUNCTION = nn.ReLU()

        self.conv_3_64 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv_64_32 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.conv_64_64 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv_32_128 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3)

        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.max_pool_odd = nn.MaxPool2d(kernel_size=4, stride=4, padding=(1, 1))

        self.fully_connected_one = nn.Linear(in_features=128, out_features=128)
        self.fully_connected_two = nn.Linear(in_features=128, out_features=128)
        self.fully_connected_three = nn.Linear(in_features=128, out_features=64)
        self.fully_connected_four = nn.Linear(in_features=64, out_features=3)

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        self.layers = nn.Sequential(
            self.conv_3_64,
            ACTIVATION_FUNCTION,
            self.max_pool_odd,
            self.conv_64_64,
            ACTIVATION_FUNCTION,
            self.max_pool,
            self.conv_64_32,
            ACTIVATION_FUNCTION,
            self.max_pool,
            self.conv_32_128,
            ACTIVATION_FUNCTION,
            self.max_pool,
            self.flatten,
            self.fully_connected_one,
            ACTIVATION_FUNCTION,
            self.fully_connected_two,
            ACTIVATION_FUNCTION,
            self.fully_connected_three,
            ACTIVATION_FUNCTION,
            self.fully_connected_four,
        )

    def forward(self, input):
        return self.layers(input)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        training_data: DataLoader,
        test_data: DataLoader,
        learning_rate: int = 0.001,
        loss_func=nn.CrossEntropyLoss(),
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        reg_param: float = 1e-5,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.training_data = training_data
        self.test_data = test_data
        self.optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=reg_param)
        self.loss_function = loss_func

    def train_neural_network(self, epoch: int) -> tuple[nn.Module, list[int]]:
        self.model.train()
        train_losses = []
        test_losses = []

        print("\nBeginning Training:\n")
        for i in range(epoch):
            correct = 0
            epoch_loss = 0
            for features, labels in self.training_data:
                features, labels = features.to(self.device), labels.to(self.device)
                model_outputs = self.model(features)
                _, predicted = torch.max(model_outputs, dim=1)

                correct += (predicted == labels).sum().item()

                batch_loss = self.loss_function(model_outputs, labels)
                epoch_loss += batch_loss.item()

                self.optimizer.zero_grad()

                batch_loss.backward()

                self.optimizer.step()

            accuracy = 100 * correct / len(self.training_data.dataset)
            epoch_loss /= len(self.training_data)
            test_accuracy, test_loss, _ = self.evaluate()
            train_losses.append(epoch_loss)
            test_losses.append(test_loss)
            print(f"Epoch {i+1}/{epoch}\tTraining Loss: {epoch_loss:.4f}\tTraining Accuracy: {accuracy:.2f}%\tTest Loss: {test_loss:.4f}\tTest Accuracy: {test_accuracy:.2f}%")

        return self.model, train_losses, test_losses

    def evaluate(self, detailed: bool = False, classes: dict[str, int] = None) -> tuple[float, float, dict[str, float]]:
        self.model.eval()
        details = {}

        if detailed:
            for cls in classes.values():
                details[cls] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

        with torch.no_grad():
            correct = 0
            average_loss = 0
            for features, labels in self.test_data:
                features, labels = features.to(self.device), labels.to(self.device)

                model_outputs = self.model(features)
                _, predicted = torch.max(model_outputs, dim=1)

                batch_loss = self.loss_function(model_outputs, labels)
                average_loss += batch_loss.item()

                correct += (predicted == labels).sum().item()

                if detailed:
                    for pred, label in zip(predicted, labels):
                        for cls in classes.values():
                            if pred == cls and label == cls:
                                details[cls]["TP"] += 1
                            if pred == cls and label != cls:
                                details[cls]["FP"] += 1
                            if pred != cls and label != cls:
                                details[cls]["TN"] += 1
                            if pred != cls and label == cls:
                                details[cls]["FN"] += 1

            accuracy = 100 * correct / len(self.test_data.dataset)
            average_loss /= len(self.test_data)
            self.model.train()
            return accuracy, average_loss, details


def visualize_losses(losses: list[float], title: str):
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(len(losses)), losses)
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")  # use cpu only if better hardware not available
    torch.manual_seed(42)
    print("Device Initialized")

    model = GuavaDiseaseClassifier()
    print("Model Initialized")

    training_data, test_data = load_guava_dataset()
    print("Dataset Loaded")

    trainer = Trainer(model, device, training_data, test_data)

    model, train_loss, test_loss = trainer.train_neural_network(epoch=17)

    accuracy, loss, details = trainer.evaluate(detailed=True, classes=CLASSES)

    print("\nTest Details:\n")
    print(f"Accuracy: {accuracy:.2f}%\nLoss: {loss:.4f}\n")
    for cls, det in details.items():
        precision = det["TP"] / (det["TP"] + det["FP"])
        recall = det["TP"] / (det["TP"] + det["FN"])
        print(f"{CLASSES_NAMES[cls]}:\tPrecision: {precision:.4f}\tRecall: {recall:.4f}")

    # save the trained model if needed
    # model_path = "pretrained_cnn.pth"
    # torch.save(model.state_dict(), model_path)
    # print(f"\nTrained model saved as {model_path}")

    visualize_losses(train_loss, "Training Loss Over Epochs")
    visualize_losses(test_loss, "Test Loss Over Epochs")


if __name__ == "__main__":
    main()
