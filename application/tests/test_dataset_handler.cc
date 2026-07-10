#include <cstdio>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "../dataset/dataset_handler.hh"
#include <filesystem>
#include <random>

// Helper: write a temporary label file and return its path
static std::string write_temp_label(const std::string& content) {
    // Generate a unique filename in the system temp directory
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 1'000'000);

    std::filesystem::path path = std::filesystem::temp_directory_path() /
                    ("test_label_" + std::to_string(dist(gen)) + ".txt");

    std::ofstream file(path);
    file << content;
    file.close();

    return path.string();
}

TEST(LoadLabelsTest, SingleLabelConversion) {
    std::string path = write_temp_label("0 0.5 0.5 0.2 0.2\n");

    std::vector<Label> labels = load_labels(path, 100, 100);
    std::filesystem::remove(path);

    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels[0].class_id, 0);
    EXPECT_EQ(labels[0].x1, 40);
    EXPECT_EQ(labels[0].y1, 40);
    EXPECT_EQ(labels[0].x2, 60);
    EXPECT_EQ(labels[0].y2, 60);
}

TEST(LoadLabelsTest, MultipleLabels) {
    std::string path = write_temp_label(
        "0 0.5 0.5 0.2 0.2\n"
        "7 0.25 0.25 0.1 0.1\n");

    std::vector<Label> labels = load_labels(path, 200, 200);
    std::filesystem::remove(path);

    ASSERT_EQ(labels.size(), 2u);
    EXPECT_EQ(labels[0].class_id, 0);
    EXPECT_EQ(labels[1].class_id, 7);
    EXPECT_EQ(labels[1].x1, 40);
    EXPECT_EQ(labels[1].y1, 40);
    EXPECT_EQ(labels[1].x2, 60);
    EXPECT_EQ(labels[1].y2, 60);
}

TEST(LoadLabelsTest, NonRectangularImageDimensions) {
    std::string path = write_temp_label("0 0.5 0.5 0.5 0.5\n");

    std::vector<Label> labels = load_labels(path, 960, 720);
    std::filesystem::remove(path);

    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels[0].x1, 240);
    EXPECT_EQ(labels[0].y1, 180);
    EXPECT_EQ(labels[0].x2, 720);
    EXPECT_EQ(labels[0].y2, 540);
}

TEST(LoadLabelsTest, MissingFileReturnsEmpty) {
    std::vector<Label> labels = load_labels("/nonexistent/path/label.txt", 100, 100);
    EXPECT_TRUE(labels.empty());
}