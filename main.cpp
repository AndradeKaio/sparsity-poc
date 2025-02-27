#include <chrono>
#include <iostream>
#include <omp.h>
#include <ostream>
#include <stdio.h>
#include <taco.h>
#include <vector>

using namespace std;
using namespace taco;

using DenseMatrix = vector<vector<double>>;

DenseMatrix genMatrix(int rows, int cols, float sparsity) {
  DenseMatrix matrix(rows, vector<double>(cols, 0.0));
  srand(time(0));

  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      if ((rand() / (double)RAND_MAX) > sparsity)
        matrix[i][j] = (rand() % 10) + 1;

  return matrix;
}

bool sampling(DenseMatrix input, float sparsity) {
  int rows = input.size();
  int cols = input.empty() ? 0 : input[0].size();
  int count = 0;
#pragma omp parallel for reduction(+ : count) collapse(2)
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      if (input[i][j] == 0)
        count++;
  return static_cast<double>(count) / (rows * cols) >= sparsity;
}

Tensor<double> convertToTACO(DenseMatrix matrix, Format format) {
  int rows = matrix.size();
  int cols = matrix.empty() ? 0 : matrix[0].size();
  Tensor<double> tensor({rows, cols}, sparse);

  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) {
      double x = matrix[i][j];
      if (x != 0)
        tensor.insert({i, j}, x);
    }
  tensor.pack();
  return tensor;
}

DenseMatrix matrixMultiply(DenseMatrix A, DenseMatrix B) {
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

Tensor<double> matrixMultiply(Tensor<double> A, Tensor<double> B,
                              Tensor<double> C) {
  IndexVar i("i"), j("j"), k("k");
  A(i, j) = sum(k, B(i, k) * C(k, j));
  A.evaluate();
  return A;
}

int main(int argc, char *argv[]) {
  int N = 1024;
  DenseMatrix b = genMatrix(N, N, 0.4);
  DenseMatrix c = genMatrix(N, N, 0.4);
  auto start = std::chrono::high_resolution_clock::now();
  cout << sampling(b, 0.3) << endl;
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
  Tensor<double> B = convertToTACO(b, Format({Sparse, Dense}));
  Tensor<double> C = convertToTACO(c, Format({Sparse, Dense}));
  Tensor<double> A({N, N}, Format({Sparse, Dense}));
  A = matrixMultiply(A, B, C);
  // cout << A << endl;
  return 0;
}
