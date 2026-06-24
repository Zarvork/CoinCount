import os

import cv2
import kagglehub
import numpy as np


def show_image(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_dataset():
    # Download and Load dataset

    dataset_path = kagglehub.dataset_download("janstaffa/euro-coins-dataset")

    print("Dataset downloaded in: ", dataset_path)

    image_dir_path = os.path.join(dataset_path, "images")

    filenames = sorted(os.listdir(image_dir_path))

    images = []

    for filename in filenames:
        img_path = os.path.join(image_dir_path, filename)

        images.append(cv2.imread(img_path))

    print(f"Number of images : {len(images)} images.")

    return images


def preprocess(image, gaussian_kernel_size):
    # Convert to GrayScale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_image = clahe.apply(gray_image)

    # Smooth to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, gaussian_kernel_size, 0)

    return blurred_image


def detect_circles(image):
    GAUSSIAN_KERNEL_SIZE = (31, 31)
    MIN_DIST = 60
    MIN_RADIUS = 30
    MAX_RADIUS = 155

    # Copy the image
    image = image.copy()

    # Image processing
    preprocessed_image = preprocess(image, GAUSSIAN_KERNEL_SIZE)

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


def draw_results(image, circles):
    image = image.copy()

    for cx, cy, r in circles:
        cx, cy, r = int(cx), int(cy), int(r)
        cv2.circle(image, (cx, cy), r, (255, 0, 0), 2)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
    return image


if __name__ == "__main__":
    # Load dataset
    images = load_dataset()

    no_detection = 0
    for ii, image in enumerate(images):
        # Coin Detection
        circles = detect_circles(image)
        if circles is not None:
            cv2.imwrite(f"output/{ii + 1}.jpg", draw_results(image, circles))
        else:
            no_detection += 1
    print(f"Images without detection: {no_detection}/{len(images)}")
