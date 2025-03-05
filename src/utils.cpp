#include <taco.h>
#include <string>
#include <fstream>

void writeKernel(const std::string& filename, const taco::Tensor<double> compiledOut) {
    using namespace std;
    ofstream file;
    file.open(filename);
    file << compiledOut.getSource();
    file.close();
}
