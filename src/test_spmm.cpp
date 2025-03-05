#include "../include/spmm.h"
#include <iostream>
#include <string>
#include <taco.h>
#include <vector>

using namespace taco;

void run(int rows, int cols, const std::string &format, double sparsity,
         bool input, const std::string &sampling, bool dense_output) {

  std::cout << "size:" << rows << "x" << cols << ", format:" << format
            << ", sparsity:" << sparsity << ", input-conversion:" << input
            << ", sampling:" << sampling << std::endl;
  if (format == "NDD") {
    DenseMatrix A = genMatrix(rows, cols, 0.0);
    DenseMatrix B = genMatrix(rows, cols, 0.0);
    ddmm(A, B);
  } else if (format == "DD") {
    DenseMatrix tmp = genMatrix(rows, cols, 0.0);
    Tensor<double> A = convertToTACO(tmp, Format({Dense, Dense}));
    tmp = genMatrix(rows, cols, 0.0);
    Tensor<double> B = convertToTACO(tmp, Format({Dense, Dense}));
    spmm(A, B, Format({Dense, Dense}));
  } else {

    Format tFormat, outFormat;
    if (format == "CSR")
      tFormat = Format({Sparse, Dense});
    else if (format == "CSC")
      tFormat = Format({Dense, Sparse});
    else {
      tFormat = Format({Sparse, Sparse}, {1, 0});
      dense_output = true;
    }

    if (dense_output)
      outFormat = Format({Dense, Dense});
    else
      outFormat = tFormat;

    DenseMatrix tmp = genMatrix(rows, cols, 0.0);
    Tensor<double> B = convertToTACO(tmp, tFormat);
    if (!input) {
      Tensor<double> A = convertToTACO(tmp, tFormat);
      if (sampling == "sampling")
        spmmSampling(A, B, outFormat, sparsity, false);
      else
        spmm(A, B, outFormat);
    } else {
      DenseMatrix A = genMatrix(rows, cols, sparsity);
      if (sampling == "sampling") {
        spmmInputSampling(A, B, outFormat, sparsity, false);
      } else if (sampling == "psampling") {
        spmmInputSampling(A, B, outFormat, sparsity, true);
      } else {
        spmmInput(A, B, outFormat);
      }
    }
  }
}

int parseArguments(int argc, char *argv[]) {
  if (argc != 8) {
    std::cerr << "Usage: " << argv[0]
              << " <rows> <cols> <format> <dense_output> <sparsity> <input> "
                 "<sampling>\n";
    return 1;
  }

  int rows = std::stoi(argv[1]);
  int cols = std::stoi(argv[2]);
  std::string format = argv[3];
  std::string dense_output_str = argv[4];
  double sparsity = std::stod(argv[5]);
  std::string input_str = argv[6];
  std::string sampling = argv[7];

  std::vector<std::string> valid_formats = {"CSR", "CSC", "DCSR", "DD", "NDD"};
  if (std::find(valid_formats.begin(), valid_formats.end(), format) ==
      valid_formats.end()) {
    std::cerr << "Invalid format: " << format << "\n";
    return 1;
  }

  bool input = (input_str == "true");
  bool dense_output = (dense_output_str == "true");

  std::vector<std::string> valid_sampling = {"none", "sampling", "psampling"};
  if (std::find(valid_sampling.begin(), valid_sampling.end(), sampling) ==
      valid_sampling.end()) {
    std::cerr << "Invalid sampling type: " << sampling << "\n";
    return 1;
  }

  run(rows, cols, format, sparsity, input, sampling, dense_output);
  return 0;
}

void runTests(int argc, char *argv[]) { parseArguments(argc, argv); }
