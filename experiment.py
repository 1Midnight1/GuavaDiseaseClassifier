import matplotlib.pyplot as plt
import torch
from cnn import load_guava_dataset, GuavaDiseaseClassifier, Trainer

lr_labels = [
    "lr=0.01",
    "lr=0.005",
    "lr=0.0025",
    "lr=0.001",
]

reg_labels = [
    "λ=1E-3",
    "λ=1E-4",
    "λ=1E-5",
    "λ=0",
]

optim_labels = [
    "SGD",
    "Adam",
]


def experiment_learning_rate() -> tuple[list[list[float]], list[list[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")  # use cpu only if better hardware not available
    torch.manual_seed(42)
    values_to_test = [0.01, 0.005, 0.0025, 0.001]
    test_losses = []
    train_losses = []
    training_data, test_data = load_guava_dataset()
    print("Dataset Loaded")
    for lr in values_to_test:
        print(f"\nExperimenting with learning rate {lr}:")
        model = GuavaDiseaseClassifier()
        print("Model Initialized")

        trainer = Trainer(model, device, training_data, test_data, learning_rate=lr)

        model, train_loss, test_loss = trainer.train_neural_network(epoch=17)
        test_losses.append(train_loss)
        train_losses.append(test_loss)

    return train_losses, test_losses


def experiment_regularization() -> tuple[list[list[float]], list[list[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")  # use cpu only if better hardware not available
    torch.manual_seed(42)
    values_to_test = [1e-3, 1e-4, 1e-5, 0]
    test_losses = []
    train_losses = []
    training_data, test_data = load_guava_dataset()
    print("Dataset Loaded")
    for lambda_reg in values_to_test:
        print(f"\nExperimenting with regularization parameter {lambda_reg}:")
        model = GuavaDiseaseClassifier()
        print("Model Initialized")

        trainer = Trainer(model, device, training_data, test_data, reg_param=lambda_reg)

        model, train_loss, test_loss = trainer.train_neural_network(epoch=17)
        test_losses.append(train_loss)
        train_losses.append(test_loss)

    return train_losses, test_losses


def experiment_optimizers() -> tuple[list[list[float]], list[list[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")  # use cpu only if better hardware not available
    torch.manual_seed(42)
    values_to_test = [
        torch.optim.SGD,
        torch.optim.torch.optim.Adam,
    ]
    test_losses = []
    train_losses = []
    training_data, test_data = load_guava_dataset()
    print("Dataset Loaded")
    for optim in values_to_test:
        print(f'\nExperimenting with optimizer "{optim}":')
        model = GuavaDiseaseClassifier()
        print("Model Initialized")

        trainer = Trainer(model, device, training_data, test_data, optimizer=optim)

        model, train_loss, test_loss = trainer.train_neural_network(epoch=17)
        test_losses.append(train_loss)
        train_losses.append(test_loss)

    return train_losses, test_losses


def plot_losses(losses: list[list[float]], labels: list[str], title: str):
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    for loss, label in zip(losses, labels):
        plt.plot(range(len(loss)), loss, label=label)
    plt.legend()

    plt.show()


# note: experiments take very long to run
if __name__ == "__main__":

    lr_train_losses, lr_test_losses = experiment_learning_rate()

    print(lr_train_losses)
    print(lr_test_losses)

    plot_losses(lr_train_losses, lr_labels, "Training Losses vs Epoch w/ Varying Learning Rate")
    plot_losses(lr_test_losses, lr_labels, "Test Losses vs Epoch w/ Varying Learning Rate")

    reg_train_losses, reg_test_losses = experiment_regularization()

    print(reg_train_losses)
    print(reg_test_losses)

    plot_losses(reg_train_losses, reg_labels, "Training Losses vs Epoch w/ Varying Regularization Parameter")
    plot_losses(reg_test_losses, reg_labels, "Test Losses vs Epoch w/ Varying Regularization Parameter")

    optim_train_losses, optim_test_losses = experiment_optimizers()

    print(optim_train_losses)
    print(optim_test_losses)

    plot_losses(optim_train_losses, optim_labels, "Training Losses vs Epoch w/ Varying Optimizers")
    plot_losses(optim_test_losses, optim_labels, "Test Losses vs Epoch w/ Varying Optimizers")
