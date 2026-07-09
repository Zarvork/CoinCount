#include <iostream>
#include <string>

#include "benchmark/benchmark.hh"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path>" << std::endl;
        return 1;
    }
    std::string dataset_path = argv[1];

    compute_metrics(dataset_path, true);

    return 0;
}