import os

import cv2
import numpy as np
from dataset_handler import CLASS_VALUES, load_dataset

GAUSSIAN_KERNEL_SIZE = (31, 31)
MIN_DIST = 60
MIN_RADIUS = 30
MAX_RADIUS = 155


def show_image(img: np.ndarray, title: str) -> None:
    """Show an image

    Args:
        img (np.ndarray): The image we want to show
        title (str): Title of the image
    """
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bb_intersection_over_union(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    """Compute IoU metric

    Args:
        box_a (tuple[float, float, float, float]): Predicted bounding box
        box_b (tuple[float, float, float, float]):  Ground Truth bounding box

    Returns:
        float: IoU value
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    interArea = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    iou = interArea / float(box_a_area + box_b_area - interArea)

    return iou


def preprocess(image: np.ndarray) -> np.ndarray:
    """Apply preprocessing on the input image

    Args:
        image (np.ndarray): The image we want to preprocess
        gaussian_kernel_size (tuple[int, int]): Shape of the gaussian kernel

    Returns:
        np.ndarray: Image resulting from the preprocessing
    """
    # Convert to GrayScale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_image = clahe.apply(gray_image)

    # Smooth to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, GAUSSIAN_KERNEL_SIZE, 0)

    return blurred_image


def detect_circles(image: np.ndarray) -> np.ndarray | None:
    """Detect the circles in the image for coin detection

    Args:
        image (np.ndarray): Image with coins

    Returns:
        np.ndarray | None: List of tuple of the form (center_x, center_y, radius)
    """

    # Image processing
    preprocessed_image = preprocess(image)

    # Coin Detection
    # Canny Edge Detection + Detect circles (Hough circle transform)
    circles = cv2.HoughCircles(
        preprocessed_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=MIN_DIST,
        param1=50,
        param2=30,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )

    if circles is None:
        return None

    circles = np.squeeze(circles, axis=0)

    return circles


def draw_results(
    image: np.ndarray,
    circles: np.ndarray,
    labels: list[tuple[int, int, int, int, int]] | None = None,
) -> np.ndarray:
    """Generate the image with predicted circles drawn and optionally ground truth bounding box

    Args:
        image (np.ndarray): Original Image with coins
        circles (np.ndarray): Predicted circles coordinates and radius
        labels (list[tuple[int, int, int, int, int]] | None, optional): Ground Truth bounding boxes. Defaults to None.

    Returns:
        np.ndarray: Image with the results and optionally ground truth boxes drawn
    """
    image = image.copy()

    # Draw predicted circles
    for cx, cy, r in circles:
        cx, cy, r = int(cx), int(cy), int(r)
        cv2.circle(image, (cx, cy), r, (255, 0, 0), 2)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

    # Draw ground truth bounding box
    if labels is not None:
        for class_id, x1, y1, x2, y2 in labels:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            value = CLASS_VALUES[class_id]
            cv2.putText(
                image,
                f"{value:.2f}E",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    return image


def iou_matching(
    labels: list[tuple[int, int, int, int, int]], circles: np.ndarray
) -> tuple[int, int, int, float]:
    """Compute scoring metrics for the given predicted circles and ground truth boxes

    Args:
        labels (list[tuple[int, int, int, int, int]]): Ground Truth bounding boxes
        circles (np.ndarray): Predicted circles coordinates and radius

    Returns:
        tuple[int, int, int, float]: Tuple with True Positive, False Positive, False Negative and IoU sum metrics
    """
    # Convert circles into bounding box
    pred_boxes = [(cx - r, cy - r, cx + r, cy + r) for cx, cy, r in circles]
    gt_boxes = [(x1, y1, x2, y2) for _, x1, y1, x2, y2 in labels]

    # Compute all IoU between all pairs of predicted and ground truth boxes
    all_pairs = []
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou = bb_intersection_over_union(pred, gt)
            all_pairs.append((iou, i, j))

    # Sort by descending order of IoU
    all_pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred = set()
    matched_gt = set()
    tp = 0
    iou_sum = 0

    for iou, i, j in all_pairs:
        if iou < 0.5:
            break  # The remaining pairs all have an IoU < 0.5 (so no good matches left)
        # Verify that the boxes are not already taken
        if i not in matched_pred and j not in matched_gt:
            iou_sum += iou
            tp += 1
            matched_pred.add(i)
            matched_gt.add(j)

    fp = len(pred_boxes) - tp  # Predicted boxes not assigned to any ground truth boxes
    fn = len(gt_boxes) - tp  # Ground Truth boxes not assigned to any predicted boxes
    return tp, fp, fn, iou_sum


def compute_metrics(save: bool = False) -> None:
    """Compute the performance of the coin detection algorithm with the EURO coins dataset

    Args:
        save (bool, optional): Save the results images in the output directory. Defaults to False.
    """
    # Load dataset
    dataset = load_dataset()

    tp_total = 0
    fp_total = 0
    fn_total = 0
    no_detection = 0
    iou_total = 0

    # Create directory to save results images
    if save:
        os.makedirs("output", exist_ok=True)
    # Iterate over each image of the dataset
    for filename, image, labels in dataset:
        # Coin Detection
        circles = detect_circles(image)
        # Verify that circles are found
        if circles is not None:
            # Save images with predicted and ground truth boxes
            if save:
                cv2.imwrite(f"output/{filename}", draw_results(image, circles, labels))
            # Compute metrics for the current image
            tp, fp, fn, iou_sum = iou_matching(labels, circles)
            if fp > 0 or fn > 0:
                print(f"{filename}: TP={tp} FP={fp} FN={fn}")
        else:
            tp, fp, fn, iou_sum = 0, 0, len(labels), 0
            no_detection += 1
        tp_total += tp
        fp_total += fp
        fn_total += fn
        iou_total += iou_sum
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    mean_iou = iou_total / tp_total if tp_total > 0 else 0
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"f1: {f1}")
    print(f"Mean IoU: {mean_iou}")
    print(f"Images without detection: {no_detection}/{len(dataset)}")


if __name__ == "__main__":
    compute_metrics(save=True)
