#include <gtest/gtest.h>

#include "../recognition/build_training_set.hh"

// Helper functions
static cv::Vec3f circle(float cx, float cy, float r) {
    return cv::Vec3f(cx, cy, r);
}

static Label make_label(int class_id, int x1, int y1, int x2, int y2) {
    return Label{class_id, x1, y1, x2, y2};
}

TEST(MatchCirclesTest, PropagatesCorrectClassId) {
    std::vector<cv::Vec3f> circles = {circle(50, 50, 10)};
    std::vector<Label> labels = {make_label(5, 40, 40, 60, 60)};

    auto matches = match_circles_to_labels(labels, circles);
    ASSERT_EQ(matches.size(), 1u);
    EXPECT_EQ(matches[0].class_id, 5);
    EXPECT_FLOAT_EQ(matches[0].cx, 50);
    EXPECT_FLOAT_EQ(matches[0].r, 10);
}

TEST(MatchCirclesTest, OneToOneMatching) {
    std::vector<cv::Vec3f> circles = {circle(50, 50, 10), circle(52, 52, 10)};
    std::vector<Label> labels = {make_label(0, 40, 40, 60, 60)};

    auto matches = match_circles_to_labels(labels, circles);
    EXPECT_EQ(matches.size(), 1u);
}

TEST(MatchCirclesTest, NoMatchBelowThreshold) {
    std::vector<cv::Vec3f> circles = {circle(500, 500, 10)};
    std::vector<Label> labels = {make_label(0, 40, 40, 60, 60)};

    auto matches = match_circles_to_labels(labels, circles);
    EXPECT_TRUE(matches.empty());
}

TEST(MatchCirclesTest, MultipleDistinctMatches) {
    std::vector<cv::Vec3f> circles = {circle(50, 50, 10), circle(200, 200, 20)};
    std::vector<Label> labels = {
        make_label(0, 40, 40, 60, 60),
        make_label(7, 180, 180, 220, 220),
    };
    auto matches = match_circles_to_labels(labels, circles);
    ASSERT_EQ(matches.size(), 2u);
    std::set<int> class_ids = {matches[0].class_id, matches[1].class_id};
    EXPECT_TRUE(class_ids.count(0) == 1);
    EXPECT_TRUE(class_ids.count(7) == 1);
}

TEST(AugmentImageTest, ZeroAngleIsIdentity) {
    cv::Mat image(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    image.at<cv::Vec3b>(0, 0) = cv::Vec3b(255, 255, 255);
    cv::Mat rotated = augment_image(image, 0);
    EXPECT_EQ(rotated.at<cv::Vec3b>(0, 0), cv::Vec3b(255, 255, 255));
}

TEST(AugmentImageTest, OutputKeepsSameSize) {
    cv::Mat image(20, 20, CV_8UC3, cv::Scalar(50, 50, 50));
    cv::Mat rotated = augment_image(image, 90);
    EXPECT_EQ(rotated.rows, 20);
    EXPECT_EQ(rotated.cols, 20);
}