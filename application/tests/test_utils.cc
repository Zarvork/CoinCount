#include <cmath>

#include <gtest/gtest.h>

#include "../utils/utils.hh"
#include "../dataset/dataset_handler.hh"

// Tests for bb_intersection_over_union

// Identical boxes
TEST(IoUTest, PerfectOverlap) {
    BoundingBox a = {0, 0, 10, 10};
    EXPECT_DOUBLE_EQ(bb_intersection_over_union(a, a), 1.0);
}

// Disjoint boxes
TEST(IoUTest, NoOverlap) {
    BoundingBox a = {0, 0, 10, 10};
    BoundingBox b = {20, 20, 30, 30};
    EXPECT_DOUBLE_EQ(bb_intersection_over_union(a, b), 0.0);
}

// Boxes with partial overlap
TEST(IoUTest, PartialOverlap) {
    BoundingBox a = {0, 0, 10, 10};
    BoundingBox b = {5, 5, 15, 15};
    EXPECT_NEAR(bb_intersection_over_union(a, b), 25.0 / 175.0, 1e-9);
}

// Boxes sharing only an edge
TEST(IoUTest, EdgeTouchingIsZero) {
    BoundingBox a = {0, 0, 10, 10};
    BoundingBox b = {10, 0, 20, 10};
    EXPECT_DOUBLE_EQ(bb_intersection_over_union(a, b), 0.0);
}

// Verify that IoU is symetric
TEST(IoUTest, IsSymmetric) {
    BoundingBox a = {0, 0, 10, 10};
    BoundingBox b = {5, 5, 15, 15};
    EXPECT_DOUBLE_EQ(bb_intersection_over_union(a, b),
                     bb_intersection_over_union(b, a));
}

// Small box fully inside a large box
TEST(IoUTest, OneBoxInsideAnother) {
    BoundingBox large = {0, 0, 10, 10};
    BoundingBox small = {4, 4, 6, 6};
    EXPECT_NEAR(bb_intersection_over_union(large, small), 4.0 / 100.0, 1e-9);
}

// Tests for iou_matching

// Helper functions
static cv::Mat make_circles(const std::vector<std::array<float, 3>>& data) {
    cv::Mat circles(static_cast<int>(data.size()), 3, CV_32F);
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        circles.at<float>(i, 0) = data[i][0];
        circles.at<float>(i, 1) = data[i][1];
        circles.at<float>(i, 2) = data[i][2];
    }
    return circles;
}

static Label make_label(int class_id, int x1, int y1, int x2, int y2) {
    return Label{class_id, x1, y1, x2, y2};
}

// 2 circles perfectly matching 2 ground truth boxes
TEST(MatchingTest, AllPerfectMatches) {
    cv::Mat circles = make_circles({{50, 50, 10}, {200, 200, 20}});
    std::vector<Label> labels = {
        make_label(0, 40, 40, 60, 60),
        make_label(1, 180, 180, 220, 220),
    };

    MatchingResult res = iou_matching(labels, circles);
    EXPECT_EQ(res.tp, 2);
    EXPECT_EQ(res.fp, 0);
    EXPECT_EQ(res.fn, 0);
}

// 2 circles, only 1 ground truth
TEST(MatchingTest, ExtraPredictionIsFalsePositive) {
    cv::Mat circles = make_circles({{50, 50, 10}, {500, 500, 10}});
    std::vector<Label> labels = {
        make_label(0, 40, 40, 60, 60),
    };

    MatchingResult res = iou_matching(labels, circles);
    EXPECT_EQ(res.tp, 1);
    EXPECT_EQ(res.fp, 1);
    EXPECT_EQ(res.fn, 0);
}

// 1 circle, 2 ground truth boxes
TEST(MatchingTest, MissedGroundTruthIsFalseNegative) {
    cv::Mat circles = make_circles({{50, 50, 10}});
    std::vector<Label> labels = {
        make_label(0, 40, 40, 60, 60),  
        make_label(1, 180, 180, 220, 220),
    };

    MatchingResult res = iou_matching(labels, circles);
    EXPECT_EQ(res.tp, 1);
    EXPECT_EQ(res.fp, 0);
    EXPECT_EQ(res.fn, 1);
}

// Two overlapping circles near the same GT box
TEST(MatchingTest, TwoPredictionsCompeteForSameGroundTruth) {
    cv::Mat circles = make_circles({{50, 50, 10}, {52, 52, 10}});
    std::vector<Label> labels = {
        make_label(0, 40, 40, 60, 60),
    };

    MatchingResult res = iou_matching(labels, circles);
    EXPECT_EQ(res.tp, 1);
    EXPECT_EQ(res.fp, 1);
    EXPECT_EQ(res.fn, 0);
}

// A circle overlapping a GT box with IoU < 0.5 must not be a TP.
TEST(MatchingTest, BelowThresholdIsNotMatched) {
    cv::Mat circles = make_circles({{50, 50, 5}});
    std::vector<Label> labels = {
        make_label(0, 50, 50, 150, 150),
    };

    MatchingResult res = iou_matching(labels, circles);
    EXPECT_EQ(res.tp, 0);
    EXPECT_EQ(res.fp, 1);
    EXPECT_EQ(res.fn, 1);
}

// No circles predicted but 2 GT
TEST(MatchingTest, EmptyPredictions) {
    cv::Mat circles;
    std::vector<Label> labels = {
        make_label(0, 40, 40, 60, 60),
        make_label(1, 180, 180, 220, 220),
    };

    MatchingResult res = iou_matching(labels, circles);
    EXPECT_EQ(res.tp, 0);
    EXPECT_EQ(res.fp, 0);
    EXPECT_EQ(res.fn, 2);
}

// 2 circles, no GT
TEST(MatchingTest, EmptyGroundTruth) {
    cv::Mat circles = make_circles({{50, 50, 10}, {200, 200, 20}});
    std::vector<Label> labels;

    MatchingResult res = iou_matching(labels, circles);
    EXPECT_EQ(res.tp, 0);
    EXPECT_EQ(res.fp, 2);
    EXPECT_EQ(res.fn, 0);
}