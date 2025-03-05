#include <taco.h>
#include <string>

#ifndef UTILS
#define UTILS
void writeKernel(const std::string& filename, const taco::Tensor<double> compiledOut);
#endif
