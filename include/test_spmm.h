#ifndef TEST_SPMM_H
#define TEST_SPMM_H

#include <string>

int parseArguments(int argc, char *argv[]);

void run(int rows, int cols, const std::string &format, double sparsity,
         bool input, const std::string &sampling);

void runTests(int argc, char *argv[]);

#endif
