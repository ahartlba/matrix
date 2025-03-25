#include <iostream>

#include "matrix.h"
#include "linalg.h"
#include "statespace.h"

using SimpleMatrix::Matrix;


int main(){

    Matrix<double> m(3, 3, 0);
    m(0, 0) = 1;
    m(1, 2) = 3;
    m(2, 1) = 5;
    std::cout << m << std::endl;

    auto result_ec = LinAlg::EigenDecomposition(m.Transpose() * m);
    auto result_svd = LinAlg::SVD<double>(m);

    std::cout << result_ec;
    std::cout << result_svd;

}