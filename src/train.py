
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

for folder in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading MNIST dataset...")
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto",
        data_home=str(DATA_DIR),
    )

    X = mnist.data.astype("float32")
    y = mnist.target.astype("int64")

    print(f"Dataset shape: {X.shape}")
    print(f"Number of labels: {len(y)}")

    # Scale pixel values from 0–255 to 0–1.
    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training MLPClassifier...")
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=30,
        random_state=42,
        early_stopping=True,
        verbose=True,
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    metrics = pd.DataFrame(
        [{"Model": "MLPClassifier", "Accuracy": accuracy}]
    )
    metrics.to_csv(OUTPUTS_DIR / "model_metrics.csv", index=False)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        cmap="Blues",
        values_format="d",
    )
    plt.title("MNIST Digit Recognition - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    joblib.dump(model, MODELS_DIR / "model.joblib")

    # Save the scaler convention used for this model.
    joblib.dump(
        {"input_scale": "divide_by_255", "image_shape": (28, 28)},
        MODELS_DIR / "preprocessing.joblib",
    )

    print("\nTraining complete!")
    print(f"Model saved to: {MODELS_DIR / 'model.joblib'}")
    print(f"Metrics saved to: {OUTPUTS_DIR / 'model_metrics.csv'}")
    print(f"Confusion matrix saved to: {OUTPUTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
