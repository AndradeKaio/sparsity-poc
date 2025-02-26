#include <stdio.h>
#include <vector>
#include <taco.h>
#include <iostream>

using namespace std;
using namespace taco;

using DenseMatrix = vector<vector<double>>;


DenseMatrix genMatrix(int rows, int cols, float sparsity){
    DenseMatrix matrix(rows, vector<double>(cols, 0.0));
    srand(time(0));

    for (int i=0; i<rows; ++i)
        for (int j=0; j<cols; ++j)
            if ((rand() / (double)RAND_MAX) > sparsity)
                matrix[i][j] = (rand() % 10) + 1;

    return matrix;
}

bool sampling(DenseMatrix input, float sparsity){
  return false;
}

Tensor<double> convertToTACO(DenseMatrix matrix, Format format){
  int rows = matrix.size(); 
  int cols = matrix.empty() ? 0 : matrix[0].size();
  Tensor<double> tensor({rows, cols}, sparse);

  for (int i=0; i<rows; i++)
    for (int j=0; j<rows; j++){
      double x = matrix[i][j];
      if(x != 0)
        tensor.insert({i, j}, x);
  }
  return tensor;
}



int main (int argc, char *argv[]){
  DenseMatrix input = genMatrix(5, 5, 0.3);
  Tensor<double> t = convertToTACO(input, Format({Dense,Sparse,Sparse}));
  cout << t << endl;
  return 0;
}
