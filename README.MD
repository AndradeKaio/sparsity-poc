POC of matrix multiplication format versioning

```bash
/opt/homebrew/opt/llvm/bin/clang++ -std=c++11 -fopenmp \
                                                          -I/Users/kaio/Projects/tensor-compilers/taco/include/ \
                                                          -L/Users/kaio/Projects/tensor-compilers/taco/build/lib/ \
                                                          -Wl,-rpath,/Users/kaio/Projects/tensor-compilers/taco/build/lib/ \
                                                          -ltaco src/main.cpp src/spmm.cpp src/test_spmm.cpp -o a.out
```
