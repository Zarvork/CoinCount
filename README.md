# CoinCount

Detects euro coins in a photo, recognizes each one and computes the total. Classical computer vision only (no deep learning), built first in Python then ported to C++.

<img width="438" height="380" alt="image" src="https://github.com/user-attachments/assets/d6c37f5a-efa8-4f7a-8549-527bb533a7fe" />

## Highlights

- **Detection:** F1 = 0.97 on 150 images
- **Recognition:** ~90 % accuracy
- **Python → C++ port** with identical results
- Unit tests and CI (GitLab): lint, type check, build, tests

## How it works

**Detection:** CLAHE → Gaussian blur → circular Hough transform.
**Recognition:** color (HSV) + texture (LBP) + gradient (HOG) features → SVM.

**Stack:** C++20 · OpenCV · CMake · GoogleTest · Python · scikit-learn

## Quick start

```bash
git clone https://github.com/Zarvork/CoinCount.git
cd CoinCount/application
cmake -B build && cmake --build build
./build/coincount recognize <dataset_path> <image.jpg>
```

Dataset: [EURO coins dataset](https://www.kaggle.com/datasets/janstaffa/euro-coins-dataset) (Kaggle).

## Known limitations

Hough parameters are tuned to the dataset's resolution, and the classifier is trained on all 150 images (no train/test split), so scores are measured on seen data.
