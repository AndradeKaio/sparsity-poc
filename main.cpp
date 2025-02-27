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
  int rows = input.size(); 
  int cols = input.empty() ? 0 : input[0].size();
  int count = 0;
  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++)
      if(input[i][j] == 0)
        count++;
  return static_cast<double>(count) / (rows * cols) >= sparsity;
}

Tensor<double> convertToTACO(DenseMatrix matrix, Format format){
  int rows = matrix.size(); 
  int cols = matrix.empty() ? 0 : matrix[0].size();
  Tensor<double> tensor({rows, cols}, sparse);

  for (int i=0; i<rows; i++)
    for (int j=0; j<cols; j++){
      double x = matrix[i][j];
      if(x != 0)
        tensor.insert({i, j}, x);
  }
  return tensor;
}



int main (int argc, char *argv[]){
  DenseMatrix input = genMatrix(5, 5, 0.3);
  cout << sampling(input, 0.3) << endl;
  Tensor<double> t = convertToTACO(input, Format({Dense,Sparse,Sparse}));
  cout << t << endl;
  return 0;
}
