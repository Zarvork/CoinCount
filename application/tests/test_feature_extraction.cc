#include <cmath>
#include <numeric>

#include <gtest/gtest.h>

#include "../recognition/feature_extraction.hh"

// Tests for extract_coin_mask

TEST(CoinMaskTest, OutputShapeAndType) {
    cv::Mat mask = extract_coin_mask(64);
    EXPECT_EQ(mask.rows, 64);
    EXPECT_EQ(mask.cols, 64);
    EXPECT_EQ(mask.type(), CV_8UC1);
}

TEST(CoinMaskTest, CenterIsFilledCornersAreEmpty) {
    cv::Mat mask = extract_coin_mask(64);
    EXPECT_GT(mask.at<uchar>(32, 32), 0);
    EXPECT_EQ(mask.at<uchar>(0, 0), 0);
    EXPECT_EQ(mask.at<uchar>(0, 63), 0);
    EXPECT_EQ(mask.at<uchar>(63, 0), 0);
    EXPECT_EQ(mask.at<uchar>(63, 63), 0);
}

// Tests for extract_coin_crop

TEST(CoinCropTest, OutputIsResizedToCropSize) {
    cv::Mat image(200, 200, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat crop = extract_coin_crop(image, 100, 100, 30);
    EXPECT_EQ(crop.rows, CROP_SIZE);
    EXPECT_EQ(crop.cols, CROP_SIZE);
}

TEST(CoinCropTest, ClampsToImageBorders) {
    cv::Mat image(200, 200, CV_8UC3, cv::Scalar(50, 50, 50));
    cv::Mat crop = extract_coin_crop(image, 5, 5, 30);
    EXPECT_EQ(crop.rows, CROP_SIZE);
    EXPECT_EQ(crop.cols, CROP_SIZE);
}

TEST(CoinCropTest, DegenerateRegionReturnsEmpty) {
    cv::Mat image(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat crop = extract_coin_crop(image, -100, -100, 10);
    EXPECT_TRUE(crop.empty());
}

// Tests for extract_size_feature

TEST(SizeFeatureTest, NormalizesByDiagonal) {
    auto feat = extract_size_feature(50.0, 300, 400);
    ASSERT_EQ(feat.size(), 1u);
    EXPECT_NEAR(feat[0], 0.1f, 1e-6);
}

TEST(SizeFeatureTest, FallbackWhenInvalidDimensions) {
    auto feat = extract_size_feature(25.0, 0, 0);
    ASSERT_EQ(feat.size(), 1u);
    EXPECT_NEAR(feat[0], 50.0f, 1e-6);
}

// Tests for bgr_to_gray

TEST(BgrToGrayTest, PureColorsLuminance) {
    cv::Mat blue(1, 1, CV_8UC3, cv::Scalar(255, 0, 0));
    cv::Mat green(1, 1, CV_8UC3, cv::Scalar(0, 255, 0));
    cv::Mat red(1, 1, CV_8UC3, cv::Scalar(0, 0, 255));

    EXPECT_EQ(bgr_to_gray(blue).at<uchar>(0, 0),
              cv::saturate_cast<uchar>(0.114 * 255));
    EXPECT_EQ(bgr_to_gray(green).at<uchar>(0, 0),
              cv::saturate_cast<uchar>(0.587 * 255));
    EXPECT_EQ(bgr_to_gray(red).at<uchar>(0, 0),
              cv::saturate_cast<uchar>(0.299 * 255));
}

TEST(BgrToGrayTest, WhiteStaysWhite) {
    cv::Mat white(1, 1, CV_8UC3, cv::Scalar(255, 255, 255));
    EXPECT_EQ(bgr_to_gray(white).at<uchar>(0, 0), 255);
}

// Tests for bilinear_interpolate  (expose in .hh — see note)

TEST(BilinearTest, ExactPixelReturnsValue) {
    cv::Mat gray = (cv::Mat_<uchar>(2, 2) << 10, 20, 30, 40);
    EXPECT_NEAR(bilinear_interpolate(gray, 0, 0), 10.0, 1e-9);
    EXPECT_NEAR(bilinear_interpolate(gray, 1, 1), 40.0, 1e-9);
}

TEST(BilinearTest, MidpointIsAverage) {
    cv::Mat gray = (cv::Mat_<uchar>(2, 2) << 10, 20, 30, 40);
    EXPECT_NEAR(bilinear_interpolate(gray, 0.5, 0.5), 25.0, 1e-9);
}

TEST(BilinearTest, OutOfBoundsIsZero) {
    cv::Mat gray = (cv::Mat_<uchar>(2, 2) << 10, 20, 30, 40);
    EXPECT_NEAR(bilinear_interpolate(gray, -5, -5), 0.0, 1e-9);
}

// Tests for extract_lbp_features

TEST(LbpFeatureTest, HistogramSize) {
    cv::Mat crop(CROP_SIZE, CROP_SIZE, CV_8UC3, cv::Scalar(120, 120, 120));
    auto feat = extract_lbp_features(crop);
    EXPECT_EQ(feat.size(), 36u);
}

TEST(LbpFeatureTest, EachScaleHistogramSumsToOne) {
    cv::Mat crop(CROP_SIZE, CROP_SIZE, CV_8UC3, cv::Scalar(120, 80, 200));
    auto feat = extract_lbp_features(crop);
    float sum_scale1 = std::accumulate(feat.begin(), feat.begin() + 10, 0.0f);
    float sum_scale2 = std::accumulate(feat.begin() + 10, feat.end(), 0.0f);
    EXPECT_NEAR(sum_scale1, 1.0f, 1e-4);
    EXPECT_NEAR(sum_scale2, 1.0f, 1e-4);
}

// Tests for extract_hog_features

TEST(HogFeatureTest, OutputSizeMatchesConstant) {
    cv::Mat crop(CROP_SIZE, CROP_SIZE, CV_8UC3, cv::Scalar(120, 120, 120));
    auto feat = extract_hog_features(crop);
    EXPECT_EQ(feat.size(), static_cast<size_t>(HOG_N_FEATURES));
}

TEST(HogFeatureTest, UniformImageProducesFiniteValues) {
    cv::Mat crop(CROP_SIZE, CROP_SIZE, CV_8UC3, cv::Scalar(100, 100, 100));
    auto feat = extract_hog_features(crop);
    for (float v : feat) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

// Tests for extract_features

TEST(ExtractFeaturesTest, ProducesExactlyNFeatures) {
    cv::Mat image(200, 200, CV_8UC3, cv::Scalar(120, 120, 120));
    auto feat = extract_features(image, 100, 100, 40);
    EXPECT_EQ(feat.size(), static_cast<size_t>(N_FEATURES));
}

TEST(ExtractFeaturesTest, DegenerateCircleReturnsEmpty) {
    cv::Mat image(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));
    auto feat = extract_features(image, -100, -100, 10);
    EXPECT_TRUE(feat.empty());
}