#include "feature_extraction.hh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

cv::Mat extract_coin_crop(const cv::Mat& image, float cx, float cy, float r) {
    int h = image.rows;
    int w = image.cols;
    int icx = (int)(cx);
    int icy = (int)(cy);
    int ir = (int)(r);

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
        float diagonal = std::sqrt((float)(imageH * imageH + imageW * imageW));
        return { (float)(r / diagonal) };
    }
    return { (float)(2.0 * r) };
}

std::vector<float> extract_color_features(const cv::Mat& crop, const cv::Mat& mask) {
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
        channel0.push_back((float)(px[0]));
        channel1.push_back((float)(px[1]));
        channel2.push_back((float)(px[2]));
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
        float bin_width = (float)(hi - lo) / HSV_N_BINS;

        for (float val : (ch == 0 ? channel0 : (ch == 1 ? channel1 : channel2))) {
            int bin_idx = (int)((val - lo) / bin_width);
            if (bin_idx < 0) bin_idx = 0;
            if (bin_idx >= HSV_N_BINS) bin_idx = HSV_N_BINS - 1;
            hist[bin_idx] += 1.0f;
        }

        float total = (float)(pixels.size());
        if (total > 0) {
            for (auto& h : hist) {
                h /= total;
            }
        }

        hists.insert(hists.end(), hist.begin(), hist.end());
    }

    std::vector<float> result;
    result.insert(result.end(), mean_std.begin(), mean_std.end());
    result.insert(result.end(), hists.begin(),    hists.end());
    return result;
}

cv::Mat bgr_to_gray(const cv::Mat& image) {
    CV_Assert(image.type() == CV_8UC3);
    cv::Mat gray(image.rows, image.cols, CV_8UC1);

    for (int y = 0; y < image.rows; ++y) {
        const cv::Vec3b* row = image.ptr<cv::Vec3b>(y);
        uchar* out = gray.ptr<uchar>(y);
        for (int x = 0; x < image.cols; ++x) {
            double b = row[x][0];
            double g = row[x][1];
            double r = row[x][2];
            double val = 0.114 * b + 0.587 * g + 0.299 * r;
            out[x] = cv::saturate_cast<uchar>(val);
        }
    }
    return gray;
}

double get_pixel_const_zero(const cv::Mat& gray, int row, int col) {
    if (row < 0 || row >= gray.rows || col < 0 || col >= gray.cols) {
        return 0.0;
    }
    return (double)(gray.at<uchar>(row, col));
}

double bilinear_interpolate(const cv::Mat& gray, double row, double col) {
    int r0 = (int)(std::floor(row));
    int c0 = (int)(std::floor(col));
    int r1 = r0 + 1;
    int c1 = c0 + 1;

    double fr = row - r0;
    double fc = col - c0;

    double v00 = get_pixel_const_zero(gray, r0, c0);
    double v01 = get_pixel_const_zero(gray, r0, c1);
    double v10 = get_pixel_const_zero(gray, r1, c0);
    double v11 = get_pixel_const_zero(gray, r1, c1);

    return (1 - fr) * (1 - fc) * v00
         + (1 - fr) * fc * v01
         + fr * (1 - fc) * v10
         + fr * fc * v11;
}

cv::Mat compute_uniform_lbp(const cv::Mat& gray, int nPoints, int radius) {
    cv::Mat lbp(gray.rows, gray.cols, CV_32F);

    std::vector<double> dRow(nPoints), dCol(nPoints);
    for (int i = 0; i < nPoints; ++i) {
        double angle = 2.0 * CV_PI * i / nPoints;
        dRow[i] = radius * std::sin(angle);
        dCol[i] = radius * std::cos(angle);
    }

    std::vector<int> bits(nPoints);

    for (int r = 0; r < gray.rows; ++r) {
        for (int c = 0; c < gray.cols; ++c) {
            double center = (double)(gray.at<uchar>(r, c));

            for (int i = 0; i < nPoints; ++i) {
                double sample = bilinear_interpolate(gray, r + dRow[i], c + dCol[i]);
                bits[i] = (sample >= center) ? 1 : 0;
            }

            int transitions = 0;
            for (int i = 0; i < nPoints; ++i) {
                int next = (i + 1) % nPoints;
                transitions += std::abs(bits[i] - bits[next]);
            }

            float value;
            if (transitions <= 2) {
                int sumBits = 0;
                for (int b : bits) sumBits += b;
                value = (float)(sumBits);
            } else {
                value = (float)(nPoints + 1);
            }

            lbp.at<float>(r, c) = value;
        }
    }

    return lbp;
}

std::vector<float> extract_lbp_features(const cv::Mat& crop, const cv::Mat& mask_input) {
    cv::Mat mask = mask_input.empty() ? extract_coin_mask(crop.rows) : mask_input;
    cv::Mat gray = bgr_to_gray(crop);

    std::vector<float> features;

    for (const auto& scale : LBP_SCALES) {
        int radius = scale.first;
        int nPoints = scale.second;
        int nBins = nPoints + 2;

        cv::Mat lbp = compute_uniform_lbp(gray, nPoints, radius);

        std::vector<double> hist(nBins, 0.0);
        size_t n = 0;
        for (int y = 0; y < lbp.rows; ++y) {
            const float* lrow = lbp.ptr<float>(y);
            const uchar* mrow = mask.ptr<uchar>(y);
            for (int x = 0; x < lbp.cols; ++x) {
                if (mrow[x] > 0) {
                    int bin = (int)(lrow[x]);
                    bin = std::clamp(bin, 0, nBins - 1);
                    hist[bin] += 1.0;
                    ++n;
                }
            }
        }

        for (int b = 0; b < nBins; ++b) {
            double density = (n > 0) ? hist[b] / (double)(n) : 0.0; // binWidth = 1
            features.push_back((float)(density));
        }
    }

    return features;
}

void compute_gradients(const cv::Mat& gray, cv::Mat& gRow, cv::Mat& gCol) {
    int rows = gray.rows, cols = gray.cols;
    gRow = cv::Mat::zeros(rows, cols, CV_64F);
    gCol = cv::Mat::zeros(rows, cols, CV_64F);

    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 0; x < cols; ++x) {
            gRow.at<double>(y, x) =
                (double)(gray.at<uchar>(y + 1, x)) -
                (double)(gray.at<uchar>(y - 1, x));
        }
    }
    for (int y = 0; y < rows; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            gCol.at<double>(y, x) =
                (double)(gray.at<uchar>(y, x + 1)) -
                (double)(gray.at<uchar>(y, x - 1));
        }
    }
}

std::vector<std::vector<std::vector<double>>> compute_cell_histograms(
    const cv::Mat& gRow, const cv::Mat& gCol,
    int cellRows, int cellCols, int orientations,
    int& nCellsRows, int& nCellsCols) {

    int rows = gRow.rows, cols = gRow.cols;
    nCellsRows = rows / cellRows;
    nCellsCols = cols / cellCols;

    std::vector<std::vector<std::vector<double>>> hist(
        nCellsRows, std::vector<std::vector<double>>(
                         nCellsCols, std::vector<double>(orientations, 0.0)));

    double step = 180.0 / orientations;

    for (int y = 0; y < nCellsRows * cellRows; ++y) {
        int ci = y / cellRows;
        for (int x = 0; x < nCellsCols * cellCols; ++x) {
            int cj = x / cellCols;

            double gr = gRow.at<double>(y, x);
            double gc = gCol.at<double>(y, x);
            double magnitude = std::sqrt(gr * gr + gc * gc);

            double orientation = std::atan2(gr, gc) * 180.0 / CV_PI;
            orientation = std::fmod(orientation, 180.0);
            if (orientation < 0) orientation += 180.0;

            int bin = (int)(orientation / step);
            bin = std::clamp(bin, 0, orientations - 1);

            hist[ci][cj][bin] += magnitude;
        }
    }

    double norm = (double)(cellRows * cellCols);
    for (auto& row : hist)
        for (auto& cell : row)
            for (double& v : cell)
                v /= norm;

    return hist;
}

void normalize_block_l2hys(std::vector<double>& block) {
    const double eps = 1e-5;

    double sumSq = 0.0;
    for (double v : block) sumSq += v * v;
    double denom = std::sqrt(sumSq + eps * eps);
    for (double& v : block) v /= denom;

    for (double& v : block) v = std::min(v, 0.2);

    sumSq = 0.0;
    for (double v : block) sumSq += v * v;
    denom = std::sqrt(sumSq + eps * eps);
    for (double& v : block) v /= denom;
}

std::vector<float> extract_hog_features(const cv::Mat& crop) {
    cv::Mat gray = bgr_to_gray(crop);

    cv::Mat gRow, gCol;
    compute_gradients(gray, gRow, gCol);

    int nCellsRows, nCellsCols;
    auto cellHist = compute_cell_histograms(
        gRow, gCol,
        HOG_PIXELS_PER_CELL_ROWS, HOG_PIXELS_PER_CELL_COLS,
        HOG_ORIENTATIONS, nCellsRows, nCellsCols);

    int bRow = HOG_CELLS_PER_BLOCK_ROWS;
    int bCol = HOG_CELLS_PER_BLOCK_COLS;
    int nBlocksRow = nCellsRows - bRow + 1;
    int nBlocksCol = nCellsCols - bCol + 1;

    std::vector<float> features;
    features.reserve((size_t)(nBlocksRow) * nBlocksCol * bRow * bCol * HOG_ORIENTATIONS);

    for (int r = 0; r < nBlocksRow; ++r) {
        for (int c = 0; c < nBlocksCol; ++c) {
            std::vector<double> block;
            block.reserve(bRow * bCol * HOG_ORIENTATIONS);

            for (int br = 0; br < bRow; ++br)
                for (int bc = 0; bc < bCol; ++bc)
                    for (int o = 0; o < HOG_ORIENTATIONS; ++o)
                        block.push_back(cellHist[r + br][c + bc][o]);

            normalize_block_l2hys(block);

            for (double v : block)
                features.push_back((float)(v));
        }
    }

    return features;
}

std::vector<float> extract_features(const cv::Mat& image, double cx, double cy, double r) {
    int h = image.rows, w = image.cols;
    cv::Mat crop = extract_coin_crop(image, cx, cy, r);
    if (crop.empty()) {
        return {};
    }
    cv::Mat mask = extract_coin_mask(crop.rows);

    auto sizeFeat  = extract_size_feature(r, h, w);
    auto colorFeat = extract_color_features(crop, mask);
    auto lbpFeat   = extract_lbp_features(crop, mask);
    auto hogFeat   = extract_hog_features(crop);

    std::vector<float> feat;
    feat.reserve(sizeFeat.size() + colorFeat.size() + lbpFeat.size() + hogFeat.size());

    feat.insert(feat.end(), sizeFeat.begin(), sizeFeat.end());
    feat.insert(feat.end(), colorFeat.begin(), colorFeat.end());
    feat.insert(feat.end(), lbpFeat.begin(), lbpFeat.end());
    feat.insert(feat.end(), hogFeat.begin(), hogFeat.end());

    return feat;
}

std::pair<cv::Mat, std::vector<int>> extract_features_batch(
    const cv::Mat& image, const std::vector<cv::Vec3f>& circles) {

    std::vector<std::vector<float>> featuresList;
    std::vector<int> validIndices;

    for (size_t idx = 0; idx < circles.size(); ++idx) {
        double cx = circles[idx][0], cy = circles[idx][1], r = circles[idx][2];
        auto feat = extract_features(image, cx, cy, r);
        if (!feat.empty()) {
            featuresList.push_back(std::move(feat));
            validIndices.push_back((int)(idx));
        }
    }

    if (featuresList.empty()) {
        return { cv::Mat(0, N_FEATURES, CV_32F), {} };
    }

    cv::Mat X((int)(featuresList.size()), N_FEATURES, CV_32F);
    for (int i = 0; i < X.rows; ++i)
        for (int j = 0; j < N_FEATURES; ++j)
            X.at<float>(i, j) = featuresList[i][j];

    return { X, validIndices };
}