import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from cnn import GuavaDiseaseClassifier, load_guava_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the pretrained CNN model
def load_pretrained_model(model_path="pretrained_cnn.pth"):
    model = GuavaDiseaseClassifier()
    # Map the model to the appropriate device (CPU in this case)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
    model.eval()  # Set to evaluation mode
    print(f"Loaded pretrained model from {model_path} onto CPU")
    return model


# Extract features using the CNN
def extract_features(data_loader, model, device):
    features = []
    labels = []
    with torch.no_grad():
        for images, label_batch in data_loader:
            images = images.to(device)
            # Forward pass up to the feature extraction layer, excluding fully connected layers
            embeddings = model.layers[:-4](images)
            features.append(embeddings.cpu().numpy())
            labels.extend(label_batch.numpy())
    return np.vstack(features), np.array(labels)


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print("Random Forest model trained.")
    return rf


def evaluate_random_forest(rf, X_test, y_test):
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Anthracnose", "Fruit Fly", "Healthy Guava"]))
    return accuracy


def main():
    print("Initializing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")  # use cpu only if better hardware not available
    model = load_pretrained_model().to(device=device)

    print("Loading dataset...")
    training_data, test_data = load_guava_dataset()

    print("Extracting features...")
    X_train, y_train = extract_features(training_data, model, device)
    X_test, y_test = extract_features(test_data, model, device)
    print("Feature extraction complete.")

    print("Training Random Forest classifier...")
    rf = train_random_forest(X_train, y_train)

    print("Evaluating Random Forest classifier...")
    evaluate_random_forest(rf, X_test, y_test)

    print("Done.")


if __name__ == "__main__":
    main()
