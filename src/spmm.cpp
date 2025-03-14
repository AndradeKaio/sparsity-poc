#include "../include/spmm.h"
#include "../include/chrono_timer.h"
#include "../include/utils.h"
#include <chrono>
#include <iostream>
#include <omp.h>
#include <ostream>
#include <string>
#include <taco.h>
#include <vector>

using namespace std;
using namespace taco;

using DenseMatrix = vector<vector<double>>;

const void printMatrix(const DenseMatrix &matrix) {
  int rows = matrix.size();
  int cols = matrix.empty() ? 0 : matrix[0].size();
  for (int i = 0; i < rows; ++i) {
    cout << endl;
    for (int j = 0; j < cols; ++j)
      cout << matrix[i][j] << " ";
  }
}

const DenseMatrix genMatrix(int rows, int cols, float sparsity) {
  DenseMatrix matrix(rows, vector<double>(cols, 0.0));
  srand(time(0));

  int total_elements = rows * cols;
  int nonzero_count = total_elements * (1.0 - sparsity);

  vector<int> indices(total_elements);
  for (int i = 0; i < total_elements; ++i)
    indices[i] = i;

  random_shuffle(indices.begin(), indices.end());

  for (int k = 0; k < nonzero_count; ++k) {
    int index = indices[k];
    int i = index / cols;
    int j = index % cols;
    matrix[i][j] = (rand() % 10) + 1;
  }

  return matrix;
}

const bool sampling(const DenseMatrix &input, float sparsity, bool parallel, int xStride=1, int yStride=1) {
  int rows = input.size();
  int cols = input.empty() ? 0 : input[0].size();
  int count = 0;
  if (parallel) {
#pragma omp parallel for reduction(+ : count) collapse(2)
    for (int i = 0; i < rows; i += yStride)
      for (int j = 0; j < cols; j += xStride)
        if (input[i][j] == 0)
          count++;
  } else {
    for (int i = 0; i < rows; i += yStride)
      for (int j = 0; j < cols; j += xStride)
        if (input[i][j] == 0)
          count++;
  }
  int xTotal = (cols + xStride - 1) / xStride;
  int yTotal = (rows + yStride - 1) / yStride;
  return static_cast<double>(count) / (xTotal * yTotal) >= sparsity;
}

const bool samplingTaco(Tensor<double> &input, float sparsity, bool parallel, int xStride, int yStride) {
  int rows = input.getDimension(0);
  int cols = input.getDimension(1);
  int count = 0;
  if (parallel) {
#pragma omp parallel for reduction(+ : count) collapse(2)
    for (int i = 0; i < rows; i += yStride)
      for (int j = 0; j < cols; j += xStride)
        if (input.at({i, j}) == 0)
          count++;
  } else {
    for (auto &val : input)
      if (val.first[0] % xStride == 0 && val.first[1] % yStride == 0)
          if (val.second == 0)
            count++;
  }

  int xTotal = (cols + xStride - 1) / xStride;
  int yTotal = (rows + yStride - 1) / yStride;
  return static_cast<double>(count) / (xTotal * yTotal) >= sparsity;
}

const Tensor<double> convertToTACO(DenseMatrix &matrix,
                                   const taco::Format &format) {
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

const Tensor<double> convertToFormat(const Tensor<double> &dense,
                                     const Format &format) {
  Tensor<double> sparse({dense.getDimension(0), dense.getDimension(1)}, format);
  for (auto &val : dense)
    sparse.insert(val.first.toVector(), val.second);
  sparse.pack();
  return sparse;
}

const DenseMatrix matrixMultiply(const DenseMatrix &A, const DenseMatrix &B) {
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

Tensor<double> matrixMultiply(Tensor<double> &A, const Tensor<double> &B,
                              const Tensor<double> &C) {
  IndexVar i("i"), j("j"), k("k");
  A(i, j) = sum(k, B(i, k) * C(k, j));
  A.evaluate();
  // writeKernel("kernel.cpp", A);
  return A;
}

const void spmm(const Tensor<double> &A, const Tensor<double> &B,
                const Format &format) {
  auto start = begin();

  int m = A.getDimension(0);
  int n = B.getDimension(1);
  Tensor<double> C({m, n}, format);
  C = matrixMultiply(C, A, B);
  end(start);
}

const void spmmInput(DenseMatrix &input, const Tensor<double> &B,
                     const Format &format) {
  auto start = begin();
  Tensor<double> A = convertToTACO(input, format);
  int m = A.getDimension(0);
  int n = B.getDimension(1);
  Tensor<double> C({m, n}, format);
  C = matrixMultiply(C, A, B);
  end(start);
}

const void spmmInputSampling(DenseMatrix &input, const Tensor<double> &B,
                             const Format &format, float sparsity,
                             bool parallel, int xStride, int yStride) {
  // Input has the desired sparsity
  auto start = begin();

  bool yes = sampling(input, sparsity, parallel, xStride, yStride);
  Tensor<double> A = convertToTACO(input, format);
  int m = A.getDimension(0);
  int n = B.getDimension(1);
  Tensor<double> C({m, n}, format);
  C = matrixMultiply(C, A, B);

  end(start);
}

const void spmmSampling(Tensor<double> &A, Tensor<double> &B,
                        const Format &format, float sparsity, bool parallel, int xStride, int yStride) {
  // Input has the desired sparsity
  auto start = begin();
  bool yes = samplingTaco(A, sparsity, parallel, xStride, yStride);
  B = convertToFormat(B, format);
  int m = A.getDimension(0);
  int n = B.getDimension(1);
  Tensor<double> C({m, n}, format);
  C = matrixMultiply(C, A, B);
  end(start);
}

const void ddmm(const DenseMatrix &A, const DenseMatrix &B) {
  auto start = begin();
  DenseMatrix c = matrixMultiply(A, B);
  end(start);
}

const void ddmmSampling(const DenseMatrix &A, const DenseMatrix &B,
                        const float sparsity, const bool parallel, int xStride, int yStride) {
  auto start = begin();
  bool yes = sampling(A, sparsity, parallel, xStride, yStride);
  DenseMatrix c = matrixMultiply(A, B);
  end(start);
}
