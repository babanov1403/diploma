#pragma once
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <set>
#include <unordered_set>
#include <numeric>

#include "utils.h"

class Matrix {
public:
    Matrix();
    Matrix(const std::vector<std::vector<double>>& vector);
    Matrix(const std::vector<double>& vector);
    Matrix(const std::initializer_list<std::initializer_list<double>>& init);
    Matrix(const std::initializer_list<double>& init);
    explicit Matrix(size_t rows, size_t columns);
    explicit Matrix(size_t rows);
    explicit Matrix(size_t rows, size_t columns, double val);
    Matrix(const Matrix& a);
    Matrix(Matrix&& other);

    double& operator()(int row_idx, int column_idx);
    double operator()(int row_idx, int column_idx) const;
    std::vector<double>& operator[](int idx);
    const std::vector<double>& operator[](int idx) const;

    Matrix operator*(double val) const;
    Matrix operator/(double val) const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix& operator*=(double val);
    Matrix& operator/=(double val);
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other);

    Matrix operator^(int power);
    Matrix& operator^=(int power);

    Matrix GetTransposed() const;
    static Matrix SolveLinearSystem(const Matrix& matrix, const Matrix& vector);
    Matrix GetInversed() const;
    size_t GetRows() const;
    size_t GetColumns() const;
    
    void UpdateMatrix(Matrix&&, const std::vector<int>&, const std::vector<int>&);

    Matrix GetSubMatrixByRow(const std::vector<int>&) const;
    Matrix GetSubMatrixByColumn(const std::vector<int>&) const;
    Matrix GetSubMatrix(const std::vector<int>&, const std::vector<int>&) const;

    static Matrix Stack(const Matrix& first, const Matrix& second);

    bool operator==(const Matrix&) const;
    bool operator<(const Matrix&) = delete;
    bool operator>(const Matrix&) = delete;

    friend std::ostream& operator<<(std::ostream&, const Matrix&);
    friend class PolynomMatrix;
    friend Matrix operator*(double val, const Matrix& matrix);

private:
    std::vector<std::vector<double>> vector_;
    size_t rows_;
    size_t columns_;
};

std::ostream& operator<<(std::ostream&, const Matrix&);

class PolynomMatrix {
public:
    PolynomMatrix();
    PolynomMatrix(int size);
    PolynomMatrix(const std::vector<Matrix>& coefs);
    PolynomMatrix(std::vector<Matrix>&& coefs);
    PolynomMatrix(const std::initializer_list<Matrix>& list);
    PolynomMatrix(const PolynomMatrix& p);
    PolynomMatrix(PolynomMatrix&& p);

    PolynomMatrix& operator=(const PolynomMatrix&);
    PolynomMatrix& operator=(PolynomMatrix&&);

    Matrix& operator[](int idx);
    const Matrix& operator[](int idx) const;
    PolynomMatrix& operator*(double val);
    PolynomMatrix& operator*(const PolynomMatrix& p);

    Matrix GetValue(double val);

    friend std::ostream& operator<<(std::ostream&, const PolynomMatrix&);

private:
    size_t size_;
    std::vector<Matrix> coefs_;
};