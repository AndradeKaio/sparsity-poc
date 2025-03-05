#ifndef SPMM_H
#define SPMM_H

#include <taco.h>
#include <vector>

using DenseMatrix = std::vector<std::vector<double>>;

const void printMatrix(const DenseMatrix& matrix);
const DenseMatrix genMatrix(int rows, int cols, float sparsity);
const bool sampling(const DenseMatrix& input, float sparsity, bool parallel);
const bool samplingTaco(taco::Tensor<double>& input, float sparsity, bool parallel);
const taco::Tensor<double> convertToTACO(DenseMatrix& matrix, const taco::Format& format);
const taco::Tensor<double> convertToFormat(const taco::Tensor<double>& dense,
                                     const taco::Format& format);
const DenseMatrix matrixMultiply(const DenseMatrix& A, const DenseMatrix& B);
taco::Tensor<double> matrixMultiply(taco::Tensor<double>& A,
                                    const taco::Tensor<double>& B,
                                    const taco::Tensor<double>& C);
const void spmm(const taco::Tensor<double>& A, const taco::Tensor<double>& B,
          const taco::Format& format = {taco::Dense, taco::Dense});
const void spmmInput(DenseMatrix& input, const taco::Tensor<double>& B,
               const taco::Format& format = {taco::Dense, taco::Dense});
const void spmmSampling(taco::Tensor<double>& A, taco::Tensor<double>& B,
                  const taco::Format& format = {taco::Dense, taco::Dense},
                  float sparsity = 0.8, bool parallel = false);
const void spmmInputSampling(DenseMatrix& input, const taco::Tensor<double>& B,
                       const taco::Format& format = {taco::Dense, taco::Dense},
                       float sparsity = 0.8, bool parallel = false);
const void ddmm(const DenseMatrix& A, const DenseMatrix& B);

#endif
