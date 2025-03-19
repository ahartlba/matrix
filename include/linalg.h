/**
 * Linear Algebra implementation
 *
 * Disclaimer: This file was created with the help of AI
 */

#pragma once
#pragma once

#include <algorithm>

#include "matrix.h"
#include <cmath>
#include <vector>

// Define a tolerance for floating point comparisons
constexpr double EPSILON = 1e-10;


namespace LinAlg {
template <typename T>
    void EigenDecomposition(const SimpleMatrix::Matrix<T>& A, std::vector<T>& eigenvalues, std::vector<std::vector<T>>& eigenvectors) {
        int n = A.Rows();
        SimpleMatrix::Matrix<T> V(n, n, 0);

        // Initialize V to the identity matrix
        for (int i = 0; i < n; ++i) {
            V(i, i) = 1;
        }

        SimpleMatrix::Matrix<T> A_copy = A;

        // Jacobi rotation method for eigen decomposition
        for (int iteration = 0; iteration < 100; ++iteration) {
            int p, q;
            T maxOffDiag = 0;

            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    if (std::abs(A_copy(i, j)) > maxOffDiag) {
                        maxOffDiag = std::abs(A_copy(i, j));
                        p = i;
                        q = j;
                    }
                }
            }

            if (maxOffDiag < EPSILON) break;

            T theta = 0.5 * std::atan2(2 * A_copy(p, q), A_copy(p, p) - A_copy(q, q));
            T cosTheta = std::cos(theta);
            T sinTheta = std::sin(theta);

            for (int i = 0; i < n; ++i) {
                T temp1 = A_copy(i, p);
                T temp2 = A_copy(i, q);
                A_copy(i, p) = cosTheta * temp1 - sinTheta * temp2;
                A_copy(i, q) = sinTheta * temp1 + cosTheta * temp2;
            }

            for (int i = 0; i < n; ++i) {
                T temp1 = A_copy(p, i);
                T temp2 = A_copy(q, i);
                A_copy(p, i) = cosTheta * temp1 - sinTheta * temp2;
                A_copy(q, i) = sinTheta * temp1 + cosTheta * temp2;
            }

            T temp1 = V(p, p);
            T temp2 = V(q, p);
            V(p, p) = cosTheta * temp1 - sinTheta * temp2;
            V(q, p) = sinTheta * temp1 + cosTheta * temp2;
        }

        // Copy diagonal elements as eigenvalues
        for (int i = 0; i < n; ++i) {
            eigenvalues[i] = A_copy(i, i);
        }

        // Copy eigenvectors
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                eigenvectors[i][j] = V(i, j);
            }
        }
    }

    template <typename T>
    void SVD(const SimpleMatrix::Matrix<T>& A, SimpleMatrix::Matrix<T>& U, SimpleMatrix::Matrix<T>& S, SimpleMatrix::Matrix<T>& V) {
        int m = A.Rows();
        int n = A.Cols();

        // Initialize U and V as identity matrices
        U = SimpleMatrix::Matrix<T>(m, m, 0);
        V = SimpleMatrix::Matrix<T>(n, n, 0);
        for (int i = 0; i < m; ++i) U(i, i) = 1;
        for (int i = 0; i < n; ++i) V(i, i) = 1;

        // Compute A^T * A
        SimpleMatrix::Matrix<T> AtA = A.Transpose() * A;

        std::vector<T> singularValues(n, 0);
        std::vector<std::vector<T>> vMatrix(n, std::vector<T>(n, 0));

        // Compute eigenvalues and eigenvectors of AtA (which are singular values squared and V)
        EigenDecomposition(AtA, singularValues, vMatrix);

        // Fill S with singular values and construct V from eigenvectors
        S = SimpleMatrix::Matrix<T>(m, n, 0);
        for (int i = 0; i < std::min(m, n); ++i) {
            S(i, i) = std::sqrt(std::abs(singularValues[i]));
        }

        // Copy eigenvectors into V
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                V(i, j) = vMatrix[i][j];
            }
        }

        // Compute U as A * V * S^{-1}
        SimpleMatrix::Matrix<T> S_inv(n, m, 0);
        for (int i = 0; i < std::min(m, n); ++i) {
            if (S(i, i) > EPSILON) {
                S_inv(i, i) = 1.0 / S(i, i);
            }
        }
        U = A * V * S_inv;
    }
    /* Inverse using Gaussian elimination with partial pivoting */
    template <typename T>
    SimpleMatrix::Matrix<T> Invert(const SimpleMatrix::Matrix<T>& A) {
    int n = A.Rows();
    if (A.Cols() != n) {
        throw std::invalid_argument("Matrix must be square to compute its inverse");
    }

    SimpleMatrix::Matrix<T> inv(A); // Start with a copy of A
    SimpleMatrix::Matrix<T> identity(n, n, 0);

    // Initialize identity matrix
    for (int i = 0; i < n; ++i) {
        identity(i, i) = 1;
    }

    // Perform Gaussian elimination with partial pivoting
    for (int i = 0; i < n; ++i) {
        // Find pivot
        T maxVal = std::abs(inv(i, i));
        int pivotRow = i;
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(inv(j, i)) > maxVal) {
                maxVal = std::abs(inv(j, i));
                pivotRow = j;
            }
        }

        // Swap rows if needed
        if (pivotRow != i) {
            for (int k = 0; k < n; ++k) {
                std::swap(inv(i, k), inv(pivotRow, k));
                std::swap(identity(i, k), identity(pivotRow, k));
            }
        }

        // Make diagonal element 1
        T diagElement = inv(i, i);
        if (std::abs(diagElement) < EPSILON) {
            throw std::runtime_error("Matrix is singular and cannot be inverted");
        }
        for (int k = 0; k < n; ++k) {
            inv(i, k) /= diagElement;
            identity(i, k) /= diagElement;
        }

        // Eliminate other rows
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                T factor = inv(j, i);
                for (int k = 0; k < n; ++k) {
                    inv(j, k) -= factor * inv(i, k);
                    identity(j, k) -= factor * identity(i, k);
                }
            }
        }
    }
    return identity;
}
};

