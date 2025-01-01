import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


from cnn import load_guava_dataset, CLASSES_NAMES
from rf import load_pretrained_model


def extract_cnn_features(model, dataloader, device):
    """
    Extract features from the convolutional layers of the CNN
    """
    model.eval()
    features = []
    labels = []

    def feature_hook(module, input, output):  # convultional layer
        features.append(output.view(output.size(0), -1).cpu().numpy())

    hook = model.conv_32_128.register_forward_hook(feature_hook)  # hook to the last convolutional layer

    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)

            model(images)  # forward pass to the model
            labels.append(batch_labels.numpy())

    hook.remove()

    return np.concatenate(features), np.concatenate(labels)


def train_svm_classifier(X_train, y_train, kernel="rbf"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    svm_classifier = SVC(kernel=kernel, C=1.0, random_state=42)  # svm classifier
    svm_classifier.fit(X_train_scaled, y_train)

    return svm_classifier, scaler


def evaluate_svm(svm_classifier, scaler, X_test, y_test):

    X_test_scaled = scaler.transform(X_test)  # scale the test features

    y_pred = svm_classifier.predict(X_test_scaled)  # predict the test features

    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred, target_names=list(CLASSES_NAMES.values())))

    return y_pred


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #check if cuda is available

    # model = GuavaDiseaseClassifier().to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")  # use cpu only if better hardware not available
    model = load_pretrained_model().to(device=device)  # load the cnn model

    training_dataloader, test_dataloader = load_guava_dataset()  # load the dataset

    print("Extracting CNN features...")  # extract the features from the last convolutional layer
    X_train, y_train = extract_cnn_features(model, training_dataloader, device)
    X_test, y_test = extract_cnn_features(model, test_dataloader, device)

    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")

    print("Training SVM Classifier...")  # train the svm classifier
    svm_classifier, scaler = train_svm_classifier(X_train, y_train)

    print("Evaluating SVM Classifier...")  # evaluate the svm classifier
    evaluate_svm(svm_classifier, scaler, X_test, y_test)


if __name__ == "__main__":
    main()
