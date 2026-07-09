#include "feature_extraction.hh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

cv::Mat extract_coin_crop(const cv::Mat& image, float cx, float cy, float r) {
    int h = image.rows;
    int w = image.cols;
    int icx = static_cast<int>(cx);
    int icy = static_cast<int>(cy);
    int ir = static_cast<int>(r);

    int x1 = std::max(icx - ir, 0);
    int y1 = std::max(icy - ir, 0);
    int x2 = std::min(icx + ir, w);
    int y2 = std::min(icy + ir, h);

    if (x2 <= x1 || y2 <= y1) {
        return cv::Mat();
    }

    cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
    cv::Mat crop = image(roi);

    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(CROP_SIZE, CROP_SIZE), 0, 0, cv::INTER_AREA);
    return resized;
}

cv::Mat extract_coin_mask(int size) {
    cv::Mat mask = cv::Mat::zeros(size, size, CV_8UC1);
    int center = size / 2;
    cv::circle(mask, cv::Point(center, center), center, cv::Scalar(255), -1);
    return mask;
}

std::vector<float> extract_size_feature(double r, int imageH, int imageW) {
    if (imageH > 0 && imageW > 0) {
        float diagonal = std::sqrt(static_cast<float>(imageH * imageH + imageW * imageW));
        return { (float)(r / diagonal) };
    }
    return { (float)(2.0 * r) };
}

cv::Mat extract_color_features(const cv::Mat& crop, const cv::Mat& mask) {
    cv::Mat effective_mask  = mask;
    if (mask.empty()) {
        effective_mask  = extract_coin_mask(crop.rows);
    }

    cv::Mat hsv;
    cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Vec3b> pixels;
    for (int i = 0; i < hsv.rows; ++i) {
        for (int j = 0; j < hsv.cols; ++j) {
            if (effective_mask.at<uchar>(i, j) > 0) {
                pixels.push_back(hsv.at<cv::Vec3b>(i, j));
            }
        }
    }

    std::vector<float> channel0, channel1, channel2;
    channel0.reserve(pixels.size());
    channel1.reserve(pixels.size());
    channel2.reserve(pixels.size());
    for (const auto& px : pixels) {
        channel0.push_back(static_cast<float>(px[0]));
        channel1.push_back(static_cast<float>(px[1]));
        channel2.push_back(static_cast<float>(px[2]));
    }

    auto mean_std = std::vector<float>(6);

    auto mean = [](const std::vector<float>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    };
    auto stddev = [&](const std::vector<float>& v) {
        float m = mean(v);
        float accum = 0.0f;
        for (float val : v) {
            accum += (val - m) * (val - m);
        }
        return std::sqrt(accum / v.size());
    };

    mean_std[0] = mean(channel0);
    mean_std[1] = mean(channel1);
    mean_std[2] = mean(channel2);
    mean_std[3] = stddev(channel0);
    mean_std[4] = stddev(channel1);
    mean_std[5] = stddev(channel2);

    std::vector<std::pair<int, int>> ranges = { {0, 180}, {0, 256}, {0, 256} };
    std::vector<float> hists;

    for (int ch = 0; ch < 3; ++ch) {
        std::vector<float> hist(HSV_N_BINS, 0.0f);
        int lo = ranges[ch].first;
        int hi = ranges[ch].second;
        float bin_width = static_cast<float>(hi - lo) / HSV_N_BINS;

        for (float val : (ch == 0 ? channel0 : (ch == 1 ? channel1 : channel2))) {
            int bin_idx = static_cast<int>((val - lo) / bin_width);
            if (bin_idx < 0) bin_idx = 0;
            if (bin_idx >= HSV_N_BINS) bin_idx = HSV_N_BINS - 1;
            hist[bin_idx] += 1.0f;
        }

        float total = static_cast<float>(pixels.size());
        if (total > 0) {
            for (auto& h : hist) {
                h /= total;
            }
        }

        hists.insert(hists.end(), hist.begin(), hist.end());
    }

    cv::Mat result(1, static_cast<int>(mean_std.size() + hists.size()), CV_32F);
    float* ptr = result.ptr<float>();
    std::copy(mean_std.begin(), mean_std.end(), ptr);
    std::copy(hists.begin(), hists.end(), ptr + mean_std.size());

    return result;
}