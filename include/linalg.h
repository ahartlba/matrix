/**
 * Linear Algebra implementation
 *
 * Disclaimer: This file was created with the help of AI
 */

#pragma once

#include "matrix.h"
#include <cmath>
#include <vector>

// Define a tolerance for floating point comparisons

namespace LinAlg
{
    using SimpleMatrix::Matrix;
    static double EPSILON = 1e-10;

    template <typename T>
    void SVD(const Matrix<T> &A, Matrix<T> &U, Matrix<T> &S, Matrix<T> &V)
    {
        int m = A.Rows();
        int n = A.Cols();

        // Initialize U and V as identity matrices
        U = Matrix<T>(m, m, 0);
        V = Matrix<T>(n, n, 0);
        for (int i = 0; i < m; ++i)
            U(i, i) = 1;
        for (int i = 0; i < n; ++i)
            V(i, i) = 1;

        // Compute A^T * A
        Matrix<T> AtA = A.Transpose() * A;

        std::vector<T> singularValues(n, 0);
        std::vector<std::vector<T>> vMatrix(n, std::vector<T>(n, 0));

        // Compute eigenvalues and eigenvectors of AtA (which are singular values squared and V)
        EigenDecomposition(AtA, singularValues, vMatrix);

        // Fill S with singular values and construct V from eigenvectors
        S = Matrix<T>(m, n, 0);
        for (int i = 0; i < std::min(m, n); ++i)
        {
            S(i, i) = std::sqrt(std::abs(singularValues[i]));
        }

        // Copy eigenvectors into V
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                V(i, j) = vMatrix[i][j];
            }
        }

        // Compute U as A * V * S^{-1}
        Matrix<T> S_inv(n, m, 0);
        for (int i = 0; i < std::min(m, n); ++i)
        {
            if (S(i, i) > EPSILON)
            {
                S_inv(i, i) = 1.0 / S(i, i);
            }
        }
        U = A * V * S_inv;
    }
    /* Inverse using Gaussian elimination with partial pivoting */
    template <typename T>
    Matrix<T> Invert(const Matrix<T> &A)
    {
        int n = A.Rows();
        if (A.Cols() != n)
        {
            throw std::invalid_argument("Matrix must be square to compute its inverse");
        }

        Matrix<T> inv(A); // Start with a copy of A
        Matrix<T> identity(n, n, 0);

        // Initialize identity matrix
        for (int i = 0; i < n; ++i)
        {
            identity(i, i) = 1;
        }

        // Perform Gaussian elimination with partial pivoting
        for (int i = 0; i < n; ++i)
        {
            // Find pivot
            T maxVal = std::abs(inv(i, i));
            int pivotRow = i;
            for (int j = i + 1; j < n; ++j)
            {
                if (std::abs(inv(j, i)) > maxVal)
                {
                    maxVal = std::abs(inv(j, i));
                    pivotRow = j;
                }
            }

            // Swap rows if needed
            if (pivotRow != i)
            {
                for (int k = 0; k < n; ++k)
                {
                    std::swap(inv(i, k), inv(pivotRow, k));
                    std::swap(identity(i, k), identity(pivotRow, k));
                }
            }

            // Make diagonal element 1
            T diagElement = inv(i, i);
            if (std::abs(diagElement) < EPSILON)
            {
                throw std::runtime_error("Matrix is singular and cannot be inverted");
            }
            for (int k = 0; k < n; ++k)
            {
                inv(i, k) /= diagElement;
                identity(i, k) /= diagElement;
            }

            // Eliminate other rows
            for (int j = 0; j < n; ++j)
            {
                if (j != i)
                {
                    T factor = inv(j, i);
                    for (int k = 0; k < n; ++k)
                    {
                        inv(j, k) -= factor * inv(i, k);
                        identity(j, k) -= factor * identity(i, k);
                    }
                }
            }
        }
        return identity;
    }

    template <typename T>
    struct EigenResult
    {
        Matrix<T> eigenvectors; // each column is an eigenvector
        Matrix<T> eigenvalues;  // column matrix of eigenvalues
    };

    template <typename T>
    void QRDecomposition(const Matrix<T> &A, Matrix<T> &Q, Matrix<T> &R)
    {
        int n = A.Rows();

        for (int k = 0; k < n; ++k)
        {
            for (int i = 0; i < n; ++i)
                Q(i, k) = A(i, k);

            for (int j = 0; j < k; ++j)
            {
                T dot = 0;
                for (int i = 0; i < n; ++i)
                    dot += Q(i, j) * A(i, k);
                R(j, k) = dot;
                for (int i = 0; i < n; ++i)
                    Q(i, k) -= dot * Q(i, j);
            }

            T norm = 0;
            for (int i = 0; i < n; ++i)
                norm += Q(i, k) * Q(i, k);
            norm = std::sqrt(norm);
            R(k, k) = norm;
            for (int i = 0; i < n; ++i)
                Q(i, k) /= norm;
        }
    }

    template <typename T>
    EigenResult<T> EigenDecomposition(Matrix<T> A, int maxIter = 100) {
        if (A.Rows() != A.Cols())
            throw std::invalid_argument("Matrix must be square.");

        int n = A.Rows();
        Matrix<T> Q(n, n);
        Matrix<T> R(n, n);
        Matrix<T> A_(A);
        Matrix<T> Q_(n, n);
        Matrix<T> R_(n, n);
        Matrix<T> eigenvectors = SimpleMatrix::Eye<T>(n);

        // shifted iteration for eigenvalues
        for (int iter = 0; iter < maxIter; ++iter) {
            T shift = A(n - 1, n - 1);

            // Shift A: A - shift * I
            for (int i = 0; i < n; ++i)
                A(i, i) -= shift;

            QRDecomposition(A, Q, R);
            A = R * Q;

            // Unshift A: A + shift * I
            for (int i = 0; i < n; ++i)
                A(i, i) += shift;

        }

        // unshifted iteration for eigenvectors
        for (int iter = 0; iter < maxIter; ++iter) {
            QRDecomposition(A_, Q_, R_);
            A_ = R_ * Q_;
            eigenvectors = eigenvectors * Q_;
        }

        // Normalize eigenvectors (columns)
        for (int j = 0; j < n; ++j) {
            T norm = 0;
            for (int i = 0; i < n; ++i)
                norm += eigenvectors(i, j) * eigenvectors(i, j);
            norm = std::sqrt(norm);
            for (int i = 0; i < n; ++i)
                eigenvectors(i, j) /= norm;
        }

        Matrix<T> eigenvalues(n, 1);
        for (int i = 0; i < n; ++i)
            eigenvalues(i, 0) = A(i, i);

        return { eigenvectors, eigenvalues };
    }



};

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const LinAlg::EigenResult<T> &res)
{
    os << "Eigenvalues: " << res.eigenvalues << "\n";
    os << "Eigenvectors: " << res.eigenvectors << std::endl;
    return os;
}
