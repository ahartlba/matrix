/**
 * Simple matrix implementation having basic functionalities like adding multiplying etc.
 */

#pragma once
#include <iostream>
#include <stdexcept>

enum class StorageType
{
    ROW_MAJOR,
    COL_MAJOR
};

// adding namespace to make sure no conflicts occur in different projects
namespace SimpleMatrix
{
    template <typename T>
    class Matrix
    {
    private:
        int mRows;
        int mCols;
        StorageType mStorageType;
        T *data;

    public:
        Matrix() : mRows(0), mCols(0), mStorageType(StorageType::ROW_MAJOR), data(nullptr) {}
        ~Matrix() {
            delete[] data;
            data = nullptr;
        }
        Matrix(const int nRows, const int nCols, const StorageType st = StorageType::ROW_MAJOR) : mRows(nRows), mCols(nCols), mStorageType(st)
        {
            if (nRows > 0 && nCols > 0)
                data = new T[nRows * nCols];
        }
        Matrix(const int nRows, const int nCols, T value, const StorageType st = StorageType::ROW_MAJOR) : mRows(nRows), mCols(nCols), data(nullptr), mStorageType(st)
        {
            if (nRows > 0 && nCols > 0)
            {
                data = new T[nRows * nCols];
                SetAll(value);
            }
        }

        Matrix(Matrix<T> &&mat) noexcept
        {
            mRows = mat.mRows;
            mCols = mat.mCols;
            mStorageType = mat.mStorageType;
            data = mat.data;
            mat.data = nullptr;
        }

        Matrix(const Matrix<T> &mat) noexcept
        {
            mRows = mat.mRows;
            mCols = mat.mCols;
            mStorageType = mat.mStorageType;
            data = nullptr;

            if (mRows > 0 && mCols > 0)
            {
                data = new T[mRows * mCols];
                for (size_t i = 0; i < mRows * mCols; ++i)
                    data[i] = mat.data[i];
            }
        }

        Matrix &operator=(const Matrix &m)
        {
            if (this == &m)
                return *this;

            mStorageType = m.mStorageType;
            if (mRows * mCols != m.mRows * m.mCols)
            {
                delete[] data;
                data = nullptr;

                mRows = m.mRows;
                mCols = m.mCols;

                if (mRows > 0 && mCols > 0)
                    data = new T[mRows * mCols];
            }

            for (size_t i = 0; i < mRows * mCols; i++)
                data[i] = m.data[i];

            return *this;
        }

        Matrix &operator*=(T factor)
        {
            for (int i = 0; i < mRows * mCols; i++)
                data[i] *= factor;
            return *this;
        }
        Matrix &operator/=(T factor)
        {
            for (int i = 0; i < mRows * mCols; i++)
                data[i] /= factor;
            return *this;
        }
        Matrix &operator+=(const Matrix &mat2)
        {
            if (this->mRows != mat2.mRows || this->mCols != mat2.mCols)
                throw std::invalid_argument("Sizes do not Match!");

            for (int i = 0; i < mRows * mCols; i++)
                data[i] += mat2.data[i];
            return *this;
        }

        Matrix &operator+=(const T val)
        {
            for (int i = 0; i < mRows * mCols; i++)
                data[i] += val;
            return *this;
        }
        Matrix &operator-=(const Matrix &mat2)
        {
            if (this->mRows != mat2.mRows || this->mCols != mat2.mCols)
                throw std::invalid_argument("Sizes do not Match!");

            for (int i = 0; i < mRows * mCols; i++)
                data[i] -= mat2.data[i];
            return *this;
        }

        Matrix &operator-=(const T val)
        {
            for (int i = 0; i < mRows * mCols; i++)
                data[i] -= val;
            return *this;
        }

        inline T &operator()(int i, int j)
        {
            return data[mStorageType == StorageType::ROW_MAJOR ? i * mCols + j : j * mRows + i];
        }

        inline const T &operator()(int i, int j) const
        {
            return data[mStorageType == StorageType::ROW_MAJOR ? i * mCols + j : j * mRows + i];
        }

        [[nodiscard]] int Rows() const { return mRows; }
        [[nodiscard]] int Cols() const { return mCols; }

        void SetStorage(const StorageType st) { mStorageType = st; }

        void SetAll(T val)
        {
            for (size_t i = 0; i < mRows * mCols; ++i)
                data[i] = val;
        }
        void SetDiag(T val)
        {
            auto x = std::min(mRows, mCols);
            for (size_t i = 0; i < x; ++i)
                (*this)(i, i) = val;
        }

        void ElemMult(const Matrix<T>& mat){
            if (mCols != mat.Cols() || mRows != mat.Rows())
                throw std::invalid_argument("Sizes do not Match!");
            for (size_t i=0; i < mRows; ++i)
                for (size_t j=0; j < mCols; ++j)
                    (*this)(i, j) *= mat(i, j);
        }

        void ReduceSize(const int h, const int w)
        {
            if ((h <= mRows) && (w <= mCols))
            {
                auto newData = new T[h * w];
                for (size_t i = 0; i < h; ++i)
                {
                    for (size_t j = 0; j < w; ++j)
                    {
                        newData[mStorageType == StorageType::ROW_MAJOR ? i * w + j : j * h + i] = (*this)(i, j);
                    }
                }
                delete[] data;
                data = newData;
                mRows = h;
                mCols = w;
            }
        }

        bool IsSymmetric()
        {
            if (mRows != mCols)
                return false;

            for (size_t j = 0; j < mRows; ++j)
                for (size_t i = j+1; i < mRows; ++i)
                    if (std::abs((*this)(i, j) - (*this)(j, i)) >= 1e-10)
                        return false;
            return true;
        }

        Matrix &Transposed()
        {
            std::swap(mRows, mCols);
            if (mStorageType == StorageType::ROW_MAJOR)
                mStorageType = StorageType::COL_MAJOR;
            else
                mStorageType = StorageType::ROW_MAJOR;
            return *this;
        }

        Matrix<T> Transpose() const
        {
            Matrix<T> m(*this);
            std::swap(m.mRows, m.mCols);
            if (m.mStorageType == StorageType::ROW_MAJOR)
                m.mStorageType = StorageType::COL_MAJOR;
            else
                m.mStorageType = StorageType::ROW_MAJOR;
            return m;
        }

        void Print(std::ostream &os = std::cout) const
        {
            for (int i = 0; i < mRows; ++i)
            {
                os << "[";
                for (int j = 0; j < mCols; ++j)
                    os << (*this)(i, j) << " ";
                os << "]";
                if (i < mRows - 1)
                    os << "\n";
            }
            os << "]" << std::endl;
        }
    };

    template <typename T>
    Matrix<T> operator*(const Matrix<T> &a, const Matrix<T> &b)
    {
        if (a.Cols() != b.Rows())
            throw std::invalid_argument("Sizes do not Match!");
        Matrix<T> c(a.Rows(), b.Cols(), 0);
        for (int i = 0; i < a.Rows(); i++)
            for (int j = 0; j < b.Cols(); j++)
                for (int k = 0; k < a.Cols(); k++)
                    c(i, j) += a(i, k) * b(k, j);
        return c;
    }
    template <typename T>
    Matrix<T> operator*(const Matrix<T> &a, const T val)
    {
        Matrix<T> c(a);
        c*= val;
        return c;
    }

    template <typename T>
    Matrix<T> ElemMult(const Matrix<T>& a, const Matrix<T>& b){
        if (a.Cols() != b.Cols() || a.Rows() != b.Rows())
            throw std::invalid_argument("Sizes do not Match!");
        Matrix<T> c(a.Cols(), a.Rows());
        for (size_t i=0; i < a.Rows(); ++i)
            for (size_t j=0; j < a.Cols(); ++j)
                c(i, j) = a(i, j) * b(i, j);
    }

    template <typename T>
    Matrix<T> operator+(Matrix<T> &a, Matrix<T> &b)
    {
        if (a.Rows() != b.Rows() || a.Cols() != b.Cols())
            throw std::invalid_argument("Sizes do not Match!");

        Matrix<T> c(a.Rows(), a.Cols());
        for (size_t i = 0; i < c.Rows(); ++i)
        {
            for (size_t j = 0; j < c.Cols(); ++j)
                c(i, j) = a(i, j) + b(i, j);
        }

        return c;
    }

    template <typename T>
    Matrix<T> Eye(const int n)
    {
        Matrix<T> e(n, n, 0);
        e.SetDiag(1);
        return e;
    }
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const SimpleMatrix::Matrix<T> &m)
{
    m.Print(os);
    return os;
}
