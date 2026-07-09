#include "coin_detection.hh"

// Hyperparameters
const cv::Size GAUSSIAN_KERNEL_SIZE = cv::Size(31, 31);
const int MIN_DIST = 60;
const int MIN_RADIUS = 30;
const int MAX_RADIUS = 155;

cv::Mat preprocess(const cv::Mat& image) {
    // Convert to GrayScale
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // Normalize image
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray_image, gray_image);

    // Smooth to reduce noise
    cv::Mat blurred_image;
    cv::GaussianBlur(gray_image, blurred_image, GAUSSIAN_KERNEL_SIZE, 0);

    return blurred_image;
}

cv::Mat detect_circles(const cv::Mat& image) {
    // Image processing
    cv::Mat preprocessed_image = preprocess(image);

    // Coin Detection
    // Canny Edge Detection + Detect circles (Hough circle transform)
    cv::Mat circles;
    cv::HoughCircles(
        preprocessed_image,
        circles,
        cv::HOUGH_GRADIENT,
        1,           // dp
        MIN_DIST,    // minDist
        50,          // param1
        30,          // param2
        MIN_RADIUS,  // minRadius
        MAX_RADIUS   // maxRadius
    );

    if (circles.empty()) {
        return circles;
    }
    circles = circles.reshape(1, circles.cols); // Reshape the matrix to have (N, 3) size with N = number of circles

    return circles;
}