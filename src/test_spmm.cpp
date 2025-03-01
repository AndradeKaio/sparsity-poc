#include "../include/spmm.h"
#include <iostream>
#include <sstream>
#include <string>
#include <taco.h>
#include <vector>

using namespace taco;

void run(int rows, int cols, const std::string &format, double sparsity,
         bool input, const std::string &sampling) {

  std::cout << "size:" << rows << "x" << cols << ", format:" << format
            << ", sparsity:" << sparsity << ", input-conversion:" << input
            << ", sampling:" << sampling << std::endl;
  if (format == "NDD") {
    DenseMatrix A = genMatrix(rows, cols, 0.0);
    DenseMatrix B = genMatrix(rows, cols, 0.0);
    ddmm(A, B);
  } else if (format == "DD") {
    Format tFormat({Dense, Dense});
    Tensor<double> A = convertToTACO(genMatrix(rows, cols, 0.0), tFormat);
    Tensor<double> B = convertToTACO(genMatrix(rows, cols, 0.0), tFormat);
    spmm(A, B, tFormat);
  } else {

    Format tFormat;
    if (format == "CSR")
      Format tFormat({Sparse, Dense});
    else if (format == "CSC")
      Format tFormat({Dense, Sparse});
    else
      Format tFormat({Sparse, Sparse});

    Tensor<double> B = convertToTACO(genMatrix(rows, cols, sparsity), tFormat);
    if (!input) {
      Tensor<double> A =
          convertToTACO(genMatrix(rows, cols, sparsity), tFormat);
      spmm(A, B, tFormat);
    } else {
      DenseMatrix A = genMatrix(rows, cols, sparsity);
      if (sampling == "sampling") {
        spmmInputSampling(A, B, tFormat, sparsity, false);
      } else if (sampling == "psampling") {
        spmmInputSampling(A, B, tFormat, sparsity, true);
      } else {
        spmmInput(A, B, tFormat);
      }
    }
  }
}

int parseArguments(int argc, char *argv[]) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " <rows> <cols> <format> <sparsity> <input> <sampling>\n";
    return 1;
  }

  int rows = std::stoi(argv[1]);
  int cols = std::stoi(argv[2]);
  std::string format = argv[3];
  double sparsity = std::stod(argv[4]);
  std::string input_str = argv[5];
  std::string sampling = argv[6];

  std::vector<std::string> valid_formats = {"CSR", "CSC", "DCSC", "DD", "NDD"};
  if (std::find(valid_formats.begin(), valid_formats.end(), format) ==
      valid_formats.end()) {
    std::cerr << "Invalid format: " << format << "\n";
    return 1;
  }

  bool input = (input_str == "true");

  std::vector<std::string> valid_sampling = {"none", "sampling", "psampling"};
  if (std::find(valid_sampling.begin(), valid_sampling.end(), sampling) ==
      valid_sampling.end()) {
    std::cerr << "Invalid sampling type: " << sampling << "\n";
    return 1;
  }

  run(rows, cols, format, sparsity, input, sampling);
  return 0;
}

void runTests(int argc, char *argv[]) { parseArguments(argc, argv); }
