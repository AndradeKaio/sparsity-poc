#ifndef TEST_SPMM_H
#define TEST_SPMM_H

#include <string>

int parseArguments(int argc, char *argv[]);

void run(int rows, int cols, const std::string &format, double sparsity,
         bool input, const std::string &sampling, bool dense_output, int xStride,
         int yStride);

void runTests(int argc, char *argv[]);

#endif
