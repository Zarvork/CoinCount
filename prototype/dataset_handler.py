import os

import cv2
import kagglehub
import pandas as pd

# Dict to map dataset classes with euros values
CLASS_VALUES = {0: 0.01, 1: 0.02, 2: 0.05, 3: 0.10, 4: 0.20, 5: 0.50, 6: 1.00, 7: 2.00}


def load_labels(label_path: str, width: int, height: int) -> pd.DataFrame:
    """Load a label file and returns a DataFrame with columns class_id, x1, y1, x2, y2 (top left and bottom right coordinates of bounding box)

    Args:
        label_path (str): Path to the label file containing image annotations in YOLO format
        width (int): width of the dataset images
        height (int): height of the dataset images

    Returns:
        pd.DataFrame: DataFrame with the class_id, top left and bottom right coordinates of the bounding box
    """

    # Verify that the file exists
    if not os.path.exists(label_path):
        return pd.DataFrame()

    # Read the file with image annotations in YOLO format
    labels_df = pd.read_csv(label_path, sep=" ", header=None)

    # Rename column names
    labels_df.columns = ["class_id", "center_x", "center_y", "width", "height"]

    # Compute top left coordinates of the bounding box (YOLO normalizes coordinates)
    labels_df["x1"] = (labels_df["center_x"] - labels_df["width"] / 2) * width
    labels_df["y1"] = (labels_df["center_y"] - labels_df["height"] / 2) * height

    # Compute bottom right coordinates of the bounding box (YOLO normalizes coordinates)
    labels_df["x2"] = (labels_df["center_x"] + labels_df["width"] / 2) * width
    labels_df["y2"] = (labels_df["center_y"] + labels_df["height"] / 2) * height

    # Convert the coordinates to int
    labels_df[["x1", "x2", "y1", "y2"]] = (
        labels_df[["x1", "x2", "y1", "y2"]].round().astype(int)
    )

    return labels_df[["class_id", "x1", "y1", "x2", "y2"]]


def load_dataset() -> list:
    """Load the dataset into a list of tuples

    Returns:
        list: List of tuples (filename, image, labels)
    """
    dataset_path = kagglehub.dataset_download("janstaffa/euro-coins-dataset")
    print("Dataset downloaded in:", dataset_path)

    # Define path for images and labels directories
    image_dir = os.path.join(dataset_path, "images")
    label_dir = os.path.join(dataset_path, "labels")

    filenames = sorted(
        [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    dataset = []
    for filename in filenames:
        # Define path for current image and label files
        img_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: load failed for {img_path}")
            continue

        h, w = image.shape[:2]

        # Load labels
        labels = load_labels(label_path, w, h)
        labels = list(labels.itertuples(index=False, name=None))
        dataset.append((filename, image, labels))

    print(f"Dataset loaded : {len(dataset)} images.")
    return dataset
