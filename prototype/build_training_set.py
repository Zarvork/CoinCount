import cv2
import numpy as np
import sys
import os

from dataset_handler import CLASS_VALUES, load_dataset
from coin_detection import bb_intersection_over_union, detect_circles
from feature_extraction import extract_coin_crop, extract_coin_mask, extract_size_feature, extract_color_features, extract_lbp_features, extract_hog_features

IOU_THRESHOLD = 0.5

def match_circles_to_labels(
    labels: list[tuple[int, int, int, int, int]], circles: np.ndarray
) -> list[tuple[int, int, int, int]]:
    """Associates each detected circle with its label (class_id) based on the best IoU match

    Args:
        labels (list[tuple[int, int, int, int, int]]): Ground Truth bounding boxes (class_id, x1, y1, x2, y2)
        circles (np.ndarray): Predicted circles (center_x, center_y, radius)

    Returns:
        list[tuple[int, int, int, int]]: List of tuples (class_id, cx, cy, r) for each circle
            that has found a valid match (IoU >= IOU_THRESHOLD)
    """
    pred_boxes = [(cx - r, cy - r, cx + r, cy + r) for cx, cy, r in circles]
    gt_boxes = [(x1, y1, x2, y2) for _, x1, y1, x2, y2 in labels]

    all_pairs = []
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou = bb_intersection_over_union(pred, gt)
            all_pairs.append((iou, i, j))

    all_pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred = set()
    matched_gt = set()
    matches = []

    for iou, i, j in all_pairs:
        if iou < IOU_THRESHOLD:
            break
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
            class_id = labels[j][0]
            cx, cy, r = circles[i]
            matches.append((class_id, cx, cy, r))

    return matches


def draw_matches(
    image: np.ndarray,
    labels: list[tuple[int, int, int, int, int]],
    circles: np.ndarray,
) -> np.ndarray:
    """Plots the predicted circles with the class assigned to them by the IoU matching,
    along with the ground truth bounding boxes, to visually verify that the matching is correct

    Args:
        image (np.ndarray): Original image containing the objects
        labels (list[tuple[int, int, int, int, int]]): Ground truth bounding boxes
        circles (np.ndarray): Predicted circles (center_x, center_y, radius)

     Returns:
        np.ndarray: Annotated image with the predicted circles (predicted class in blue) and the
            ground truth boxes (true class in green)
    """
    for class_id, x1, y1, x2, y2 in labels:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        value = CLASS_VALUES[class_id]
        cv2.putText(
            image,
            f"GT:{value:.2f}E",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    matches = match_circles_to_labels(labels, circles)
    for class_id, cx, cy, r in matches:
        cx, cy, r = int(cx), int(cy), int(r)
        value = CLASS_VALUES[class_id]
        cv2.circle(image, (cx, cy), r, (255, 0, 0), 2)
        cv2.putText(
            image,
            f"{value:.2f}E",
            (cx - r, cy + r + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    return image


AUGMENT_ANGLES = [0, 90, 180, 270]


def augment_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def build_training_set(augment: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Builds the training dataset (X, y) from the circles detected across the entire dataset.
    Args:
        augment (bool): Apply rotation augmentation. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: X (n_samples, n_features) and y (n_samples,) with class_id
    """
    dataset = load_dataset()
    X = []
    y = []

    for filename, image, labels in dataset:
        circles = detect_circles(image)
        if circles is None or not labels:
            continue

        matches = match_circles_to_labels(labels, circles)
        h, w = image.shape[:2]

        for class_id, cx, cy, r in matches:
            crop = extract_coin_crop(image, cx, cy, r)
            if crop is None:
                continue

            mask = extract_coin_mask(crop.shape[0])
            size_feat  = extract_size_feature(r, h, w)
            color_feat = extract_color_features(crop, mask)  # Invariant à la rotation

            angles = AUGMENT_ANGLES if augment else [0]
            for angle in angles:
                rotated = augment_image(crop, angle) if angle != 0 else crop
                lbp_feat = extract_lbp_features(rotated, mask)
                hog_feat = extract_hog_features(rotated)
                feat = np.concatenate([size_feat, color_feat, lbp_feat, hog_feat])
                X.append(feat)
                y.append(class_id)

    n_orig = 0
    if augment:
        n_orig = len(X) // len(AUGMENT_ANGLES)
    else:
        n_orig = len(X)
    print(f"Training set built: got {len(X)} samples with {n_orig} originaux x {len(AUGMENT_ANGLES)} rotations")

    return np.stack(X), np.array(y)


def visualize_matches(save_dir: str = "output_matches") -> None:
    dataset = load_dataset()
    os.makedirs(save_dir, exist_ok=True)

    for filename, image, labels in dataset:
        circles = detect_circles(image)
        if circles is None or not labels:
            continue

        annotated = draw_matches(image, labels, circles)
        cv2.imwrite(os.path.join(save_dir, filename), annotated)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "visualize":
        visualize_matches()
    else:
        X, y = build_training_set()
        np.save("X_train.npy", X)
        np.save("y_train.npy", y)
        print(f"X shape: {X.shape}, y shape: {y.shape}")