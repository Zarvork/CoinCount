import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from build_training_set import build_training_set, match_circles_to_labels
from coin_detection import detect_circles
from dataset_handler import CLASS_VALUES, load_dataset
from feature_extraction import extract_features

MODEL_PATH = "coin_classifier.joblib"
N_SPLITS = 5 


def train_classifier(X: np.ndarray, y: np.ndarray) -> Pipeline:
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", random_state=42)),
    ])

    param_grid = {
        "svm__C":     [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
    search.fit(X, y)

    print(f"Best Hyperparameters : {search.best_params_}")
    print(f"Best accuracy : {search.best_score_:.4f}")

    return search.best_estimator_


def evaluate_classification(
    pipeline: Pipeline, X: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    """Evaluates classification performance using stratified cross-validation

    Args:
        pipeline (Pipeline): Model to evaluate
        X (np.ndarray): Feature matrix
        y (np.ndarray): True labels

    Returns:
        tuple[float, np.ndarray]: Overall accuracy and cross-validated predictions
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    return acc, y_pred


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "confusion_matrix.png"):
    labels = sorted(CLASS_VALUES.keys())
    display_labels = [f"{CLASS_VALUES[c]:.2f}€" for c in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix - Part Recognition")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Real class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved at: {save_path}")


def compute_sum_error(
    X: np.ndarray, y_true: np.ndarray, y_pred_cv: np.ndarray
) -> tuple[float, list[float]]:
    """Calculates the absolute error on the total sum using cross-validated predictions.

    Args:
        X (np.ndarray): Feature matrix (not used directly, included for consistency)
        y_true (np.ndarray): True labels (class_id) in the order of the training set
        y_pred_cv (np.ndarray): Cross-validated predictions in the same order

    Returns:
        tuple[float, list[float]]: Mean absolute error and a list of errors per image
    """
    dataset = load_dataset()
    errors = []
    sample_idx = 0 

    for filename, image, labels in dataset:
        if not labels:
            continue

        circles = detect_circles(image)
        if circles is None:
            real_sum = sum(CLASS_VALUES[c] for c, *_ in labels)
            errors.append(real_sum)
            continue

        matches = match_circles_to_labels(labels, circles)
        if not matches:
            continue

        n_valid = sum(
            1 for _, cx, cy, r in matches
            if extract_features(image, cx, cy, r) is not None
        )
        if n_valid == 0 or sample_idx + n_valid > len(y_true):
            continue

        y_pred_img = y_pred_cv[sample_idx: sample_idx + n_valid]
        y_true_img = y_true[sample_idx: sample_idx + n_valid]
        sample_idx += n_valid

        predicted_sum = sum(CLASS_VALUES[c] for c in y_pred_img)
        real_sum = sum(CLASS_VALUES[c] for c in y_true_img)
        errors.append(abs(real_sum - predicted_sum))

    mean_error = float(np.mean(errors)) if errors else float("inf")
    return mean_error, errors


def run_evaluation() -> None:
    print("=== Loading/Building the Training Dataset ===")
    try:
        X = np.load("X_train.npy")
        y = np.load("y_train.npy")
    except FileNotFoundError:
        X, y = build_training_set()
        np.save("X_train.npy", X)
        np.save("y_train.npy", y)

    print("")
    print("Class distribution")
    for class_id, count in zip(*np.unique(y, return_counts=True)):
        print(f"  {CLASS_VALUES[class_id]:.2f}: {count} samples")

    print("")
    print("=== Training the SVM Classifier ===")
    pipeline = train_classifier(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Save at {MODEL_PATH}")

    print("")
    print(f"=== Evaluation - Classification ({N_SPLITS}-fold cross-validation) ===")
    accuracy, y_pred = evaluate_classification(pipeline, X, y)
    print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
    target = "> 80-85%"
    status = "Succeeded" if accuracy >= 0.80 else "Fail"
    print(f"Goal : {target} -> {status}")

    plot_confusion_matrix(y, y_pred)

    print("")
    print("=== Evaluation - Absolute error on the sum (user metric) ===")
    mean_error, errors = compute_sum_error(X, y, y_pred)
    print(f"Mean Absolute Error : {mean_error:.4f}€")
    print(f"Median error : {np.median(errors):.4f}€")
    print(f"Max error : {np.max(errors):.4f}€")
    target_err = "< 0.50€"
    status_err = "Succeeded" if mean_error < 0.50 else "Fail"
    print(f"Objectif : {target_err} -> {status_err}")


if __name__ == "__main__":
    run_evaluation()