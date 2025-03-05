#include "../include/spmm.h"
#include "../include/chrono_timer.h"
#include "../include/utils.h"
#include <chrono>
#include <string>
#include <iostream>
// #include <omp.h>
#include <ostream>
#include <taco.h>
#include <vector>

using namespace std;
using namespace taco;

using DenseMatrix = vector<vector<double>>;

const void printMatrix(const DenseMatrix& matrix) {
  int rows = matrix.size();
  int cols = matrix.empty() ? 0 : matrix[0].size();
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      cout << matrix[i][j];
}

const DenseMatrix genMatrix(int rows, int cols, float sparsity) {
  DenseMatrix matrix(rows, vector<double>(cols, 0.0));
  srand(time(0));

  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      if ((rand() / (double)RAND_MAX) > sparsity)
        matrix[i][j] = (rand() % 10) + 1;

  return matrix;
}

const bool sampling(const DenseMatrix& input, float sparsity, bool parallel) {
  int rows = input.size();
  int cols = input.empty() ? 0 : input[0].size();
  int count = 0;
  if (parallel) {
#pragma omp parallel for reduction(+ : count) collapse(2)
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        if (input[i][j] == 0)
          count++;
  } else {
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        if (input[i][j] == 0)
          count++;
  }
  return static_cast<double>(count) / (rows * cols) >= sparsity;
}

const bool samplingTaco(Tensor<double>& input, float sparsity, bool parallel) {
  int rows = input.getDimension(0);
  int cols = input.getDimension(1);
  int count = 0;
  if (parallel) {
#pragma omp parallel for reduction(+ : count) collapse(2)
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        if (input.at({i, j}) == 0)
          count++;
  } else {
    for (auto &val : input)
      if (val.second == 0)
        count++;
  }
  return static_cast<double>(count) / (rows * cols) >= sparsity;
}

const Tensor<double> convertToTACO(DenseMatrix& matrix, const taco::Format& format) {
  int rows = matrix.size();
  int cols = matrix.empty() ? 0 : matrix[0].size();
  Tensor<double> tensor({rows, cols}, format);

  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) {
      double x = matrix[i][j];
      if (x != 0)
        tensor.insert({i, j}, x);
    }
  tensor.pack();
  return tensor;
}

const Tensor<double> convertToFormat(const Tensor<double>& dense, const Format& format) {
  Tensor<double> sparse({dense.getDimension(0), dense.getDimension(1)}, format);
  for (auto &val : dense)
    sparse.insert(val.first.toVector(), val.second);
  sparse.pack();
  return sparse;
}

const DenseMatrix matrixMultiply(const DenseMatrix& A, const DenseMatrix& B) {
  int m = A.size();
  int n = B[0].size();
  int p = A[0].size();

  DenseMatrix C(m, vector<double>(n, 0.0));

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < p; k++)
        C[i][j] += A[i][k] * B[k][j];
  return C;
}

Tensor<double> matrixMultiply(Tensor<double>& A, const Tensor<double>& B,
                              const Tensor<double>& C) {
  IndexVar i("i"), j("j"), k("k");
  A(i, j) = sum(k, B(i, k) * C(k, j));
  A.evaluate();
  // writeKernel("kernel.cpp", A);
  return A;
}

const void spmm(const Tensor<double>& A, const Tensor<double>& B, const Format& format) {
  auto start = begin();

  int m = A.getDimension(0);
  int n = B.getDimension(1);
  Tensor<double> C({m, n}, format);
  C = matrixMultiply(C, A, B);
  end(start);
}

const void spmmInput(DenseMatrix& input, const Tensor<double>& B, const Format& format) {
  auto start = begin();
  Tensor<double> A = convertToTACO(input, format);
  int m = A.getDimension(0);
  int n = B.getDimension(1);
  Tensor<double> C({m, n}, format);
  C = matrixMultiply(C, A, B);
  end(start);
}

const void spmmInputSampling(DenseMatrix& input, const Tensor<double>& B, const Format& format,
                       float sparsity, bool parallel) {
  // Input has the desired sparsity
  auto start = begin();

  bool yes = sampling(input, sparsity, parallel);
  Tensor<double> A = convertToTACO(input, format);
  int m = A.getDimension(0);
  int n = B.getDimension(1);
  Tensor<double> C({m, n}, format);
  C = matrixMultiply(C, A, B);

  end(start);
}

const void spmmSampling(Tensor<double>& A, Tensor<double>& B, const Format& format,
                  float sparsity, bool parallel) {
  // Input has the desired sparsity
  auto start = begin();
  bool yes = samplingTaco(A, sparsity, parallel);
  B = convertToFormat(B, format);
  int m = A.getDimension(0);
  int n = B.getDimension(1);
  Tensor<double> C({m, n}, format);
  C = matrixMultiply(C, A, B);
  end(start);
}

const void ddmm(const DenseMatrix& A, const DenseMatrix& B) {
  auto start = begin();
  DenseMatrix c = matrixMultiply(A, B);
  end(start);
}
