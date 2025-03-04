#ifndef SPMM_H
#define SPMM_H

#include <taco.h>
#include <vector>

using DenseMatrix = std::vector<std::vector<double>>;

DenseMatrix genMatrix(int rows, int cols, float sparsity);
bool sampling(DenseMatrix input, float sparsity, bool parallel);
taco::Tensor<double> convertToTACO(DenseMatrix matrix, taco::Format format);
taco::Tensor<double> convertToFormat(taco::Tensor<double> dense, taco::Format format);
DenseMatrix matrixMultiply(DenseMatrix A, DenseMatrix B);
taco::Tensor<double> matrixMultiply(taco::Tensor<double> A,
                                    taco::Tensor<double> B,
                                    taco::Tensor<double> C);
void spmm(taco::Tensor<double> A, taco::Tensor<double> B,
          taco::Format format = {taco::Sparse, taco::Dense});
void spmmInput(DenseMatrix input, taco::Tensor<double> B,
               taco::Format format = {taco::Sparse, taco::Dense});
void spmmSampling(taco::Tensor<double> A, taco::Tensor<double> B,
                       taco::Format format = {taco::Sparse, taco::Dense},
                       float sparsity = 0.8, bool parallel = false);
void spmmInputSampling(DenseMatrix input, taco::Tensor<double> B,
                       taco::Format format = {taco::Sparse, taco::Dense},
                       float sparsity = 0.8, bool parallel = false);
void ddmm(DenseMatrix A, DenseMatrix B);

#endif
