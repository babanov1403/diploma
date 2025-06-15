#include "matrix.h"
Matrix::Matrix() : rows_(0), columns_(0) {
}
Matrix::Matrix(const std::vector<std::vector<double>>& vector)
    : vector_(vector), rows_(vector.size()), columns_(vector[0].size()) {
}
Matrix::Matrix(const std::vector<double>& vector) : rows_(1), columns_(vector.size()) {
    vector_ = std::vector<std::vector<double>>{vector};
}
Matrix::Matrix(const std::initializer_list<std::initializer_list<double>>& init) {
    rows_ = init.size();
    columns_ = (*init.begin()).size();
    for (const auto& data : init) {
        vector_.push_back(data);
    }
}
Matrix::Matrix(const std::initializer_list<double>& init)
    : vector_{init}, rows_(1), columns_(init.size()) {
}
Matrix::Matrix(size_t rows, size_t columns) : rows_(rows), columns_(columns) {
    vector_ = std::vector<std::vector<double>>(rows_, std::vector<double>(columns_));
}
Matrix::Matrix(size_t rows) : rows_(rows), columns_(rows) {
    vector_ = std::vector<std::vector<double>>(rows, std::vector<double>(rows, 0));
    for (int idx = 0; idx < rows; idx++) {
        vector_[idx][idx] = 1;
    }
}
Matrix::Matrix(size_t rows, size_t columns, double val) : rows_(rows), columns_(columns) {
    vector_ = std::vector<std::vector<double>>(rows_, std::vector<double>(columns_, val));
}
Matrix::Matrix(const Matrix& a) {
    rows_ = a.rows_;
    columns_ = a.columns_;
    vector_ = a.vector_;
}
Matrix::Matrix(Matrix&& other) {
    rows_ = other.rows_;
    other.rows_ = 0;
    columns_ = other.columns_;
    other.columns_ = 0;
    vector_ = std::move(other.vector_);
}
double& Matrix::operator()(int row_idx, int column_idx) {
    return vector_[row_idx][column_idx];
}
double Matrix::operator()(int row_idx, int column_idx) const {
    return vector_[row_idx][column_idx];
}
std::vector<double>& Matrix::operator[](int idx) {
    return vector_[idx];
}
const std::vector<double>& Matrix::operator[](int idx) const {
    return vector_[idx];
}
Matrix Matrix::operator*(double val) const {
    Matrix result = *this;
    for (auto& row : result.vector_) {
        for (auto& element : row) {
            element *= val;
        }
    }
    return result;
}
Matrix Matrix::operator/(double val) const {
    Matrix result = *this;
    return result * (1. / val);
}
Matrix Matrix::operator+(const Matrix& other) const {
    Matrix result = *this;
    for (int idx = 0; idx < rows_; idx++) {
        for (int jdx = 0; jdx < columns_; jdx++) {
            result(idx, jdx) += other(idx, jdx);
        }
    }
    return result;
}
Matrix Matrix::operator-(const Matrix& other) const {
    Matrix copy = *this;
    for (int idx = 0; idx < rows_; idx++) {
        for (int jdx = 0; jdx < columns_; jdx++) {
            copy(idx, jdx) -= other(idx, jdx);
        }
    }
    return copy;
}
Matrix Matrix::operator*(const Matrix& other) const {
    Matrix res(rows_, other.columns_);
    for (int idx = 0; idx < rows_; idx++) {
        for (int jdx = 0; jdx < other.columns_; jdx++) {
            res(idx, jdx) = 0;
            for (int kdx = 0; kdx < columns_; kdx++) {
                res(idx, jdx) += (*this)(idx, kdx) * other(kdx, jdx);
            }
        }
    }
    return res;
}
Matrix& Matrix::operator*=(double val) {
    for (auto& row : vector_) {
        for (auto& element : row) {
            element *= val;
        }
    }
    return *this;
}
Matrix& Matrix::operator/=(double val) {
    *this *= (1. / val);
    return *this;
}
Matrix& Matrix::operator+=(const Matrix& other) {
    for (int idx = 0; idx < rows_; idx++) {
        for (int jdx = 0; jdx < columns_; jdx++) {
            this->operator()(idx, jdx) += other(idx, jdx);
        }
    }
    return *this;
}
Matrix& Matrix::operator-=(const Matrix& other) {
    *this = *this - other;
    return *this;
}
Matrix& Matrix::operator*=(const Matrix& other) {
    Matrix res(rows_, other.columns_);
    for (int idx = 0; idx < rows_; idx++) {
        for (int jdx = 0; jdx < other.columns_; jdx++) {
            res(idx, jdx) = 0;
            for (int kdx = 0; kdx < columns_; kdx++) {
                res(idx, jdx) += (*this)(idx, kdx) * other(kdx, jdx);
            }
        }
    }
    *this = std::move(res);
    return *this;
}
Matrix& Matrix::operator=(const Matrix& a) {
    rows_ = a.rows_;
    columns_ = a.columns_;
    vector_ = a.vector_;
    return *this;
}
Matrix& Matrix::operator=(Matrix&& other) {
    if (this == &other) {
        return *this;
    }
    rows_ = other.rows_;
    other.rows_ = 0;
    columns_ = other.columns_;
    other.columns_ = 0;
    vector_ = std::move(other.vector_);
    return *this;
}
Matrix Matrix::operator^(int power) {
    Matrix helper(rows_);
    Matrix result = *this;
    while (power > 0) {
        if ((power & 1) != 0) {
            helper *= result;
        }
        result = result * result;
        power >>= 1;
    }
    return helper;
}
Matrix& Matrix::operator^=(int power) {
    Matrix helper(rows_);
    while (power > 0) {
        if ((power & 1) != 0) {
            helper *= *this;
        }
        *this *= *this;
        power >>= 1;
    }
    return *this;
}
Matrix Matrix::GetTransposed() const {
    Matrix transposed(columns_, rows_);
    for (int idx = 0; idx < columns_; idx++) {
        for (int jdx = 0; jdx < rows_; jdx++) {
            transposed(idx, jdx) = this->operator()(jdx, idx);
        }
    }
    return transposed;
}
Matrix Matrix::GetInversed() const {
    Matrix result(rows_, columns_, 0);
    for (int idx = 0; idx < rows_; idx++) {
        Matrix one_row(rows_, 1, 0);
        one_row(idx, 0) = 1;
        Matrix solved_system = SolveLinearSystem(*this, one_row);
        for (int jdx = 0; jdx < rows_; jdx++) {
            result(jdx, idx) = solved_system(jdx, 0);
        }
    }
    return result;
}
std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    for (auto idx : matrix.vector_) {
        for (double jdx : idx) {
            os << std::setprecision(10) << jdx << ", ";
        }
        os << '\n';
    }
    return os;
}
bool Matrix::operator==(const Matrix& other) const {
    if (other.rows_ != rows_ || other.columns_ != columns_) {
        return false;
    }
    for (int idx = 0; idx < rows_; idx++) {
        for (int jdx = 0; jdx < columns_; jdx++) {
            if (!EqualNotStrict(other.vector_[idx][jdx], vector_[idx][jdx])) {
                return false;
            }
        }
    }
    return true;
}
Matrix Matrix::SolveLinearSystem(const Matrix& matrix, const Matrix& vector) {
    size_t rows = vector.rows_;
    Matrix concat(rows, rows + 1);
    for (int idx = 0; idx < rows; idx++) {
        std::copy(matrix[idx].begin(), (matrix[idx]).end(), (concat[idx]).begin());
        concat(idx, rows) = vector(idx, 0);
    }
    for (int jdx = 0; jdx < rows; jdx++) {
        int maxx = jdx;
        for (int idx = jdx + 1; idx < rows; idx++) {
            if (std::abs(concat(maxx, jdx)) < std::abs(concat(idx, jdx))) {
                std::swap(concat[maxx], concat[idx]);
                maxx = idx;
            }
        }
        double denom = concat(jdx, jdx);
        for (int idx = 0; idx < rows + 1; idx++) {
            concat(jdx, idx) = concat(jdx, idx) * 1.0 / denom;
        }
        for (int idx = jdx + 1; idx < rows; idx++) {
            denom = concat(idx, jdx);
            for (int kdx = 0; kdx < rows + 1; kdx++) {
                concat(idx, kdx) = concat(idx, kdx) - (concat(jdx, kdx) * denom);
            }
        }
    }
    for (int jdx = static_cast<int>(rows) - 1; jdx > 0; jdx--) {
        for (int idx = jdx - 1; idx >= 0; idx--) {
            double denom = concat(idx, jdx);
            for (int kdx = 0; kdx < rows + 1; kdx++) {
                concat(idx, kdx) = concat(idx, kdx) - (concat(jdx, kdx) * denom);
            }
        }
    }
    Matrix res(rows, 1);
    for (size_t idx = 0; idx < rows; idx++) {
        res(idx, 0) = concat(idx, rows);
    }
    return res;
}
size_t Matrix::GetRows() const {
    return rows_;
}
size_t Matrix::GetColumns() const {
    return columns_;
}
void Matrix::UpdateMatrix(Matrix&& modified, const std::vector<int>& row_idxes,
                          const std::vector<int>& col_idxes) {
    for (int idx = 0; idx < row_idxes.size(); idx++) {
        for (int jdx = 0; jdx < col_idxes.size(); jdx++) {
            vector_[row_idxes[idx]][col_idxes[jdx]] = modified(idx, jdx);
        }
    }
}
Matrix Matrix::GetSubMatrixByRow(const std::vector<int>& idxes) const {
    std::vector<int> iota(columns_);
    std::ranges::iota(iota, 0);
    return GetSubMatrix(idxes, iota);
}
Matrix Matrix::GetSubMatrixByColumn(const std::vector<int>& idxes) const {
    std::vector<int> iota(rows_);
    std::ranges::iota(iota, 0);
    return GetSubMatrix(iota, idxes);
}
// maybe i should do some sort of MatrixView, we'll see
Matrix Matrix::GetSubMatrix(const std::vector<int>& idxes, const std::vector<int>& jdxes) const {
    Matrix result(idxes.size(), jdxes.size(), 0);
    int res_idx = 0;
    int res_jdx = 0;
    for (auto idx : idxes) {
        for (auto jdx : jdxes) {
            result(res_idx, res_jdx) = vector_[idx][jdx];
            ++res_jdx;
        }
        ++res_idx;
        res_jdx = 0;
    }
    return result;
}
Matrix Matrix::Stack(const Matrix& first, const Matrix& second) {
    if (first.GetColumns() != second.GetColumns()) {
        throw std::logic_error("size of matrix does not match while stacking\n");
    }
    if (first.GetRows() == 0) {
        return second;
    }
    if (second.GetRows() == 0) {
        return first;
    }
    Matrix result(first.GetRows() + second.GetRows(), first.GetColumns(), 0);
    for (int idx = 0; idx < first.GetRows() + second.GetRows(); idx++) {
        for (int jdx = 0; jdx < first.GetColumns(); jdx++) {
            result(idx, jdx) =
                idx >= first.GetRows() ? second(idx - first.GetRows(), jdx) : first(idx, jdx);
        }
    }
    return result;
}
PolynomMatrix::PolynomMatrix() : size_(0) {
}
PolynomMatrix::PolynomMatrix(int size) : size_(size) {
    coefs_ = std::vector<Matrix>(size_);
}
PolynomMatrix::PolynomMatrix(const std::vector<Matrix>& coefs)
    : coefs_(coefs), size_(coefs.size()) {
}
PolynomMatrix::PolynomMatrix(std::vector<Matrix>&& coefs)
    : coefs_(std::move(coefs)), size_(coefs.size()) {
}
PolynomMatrix::PolynomMatrix(const std::initializer_list<Matrix>& list) {
    coefs_ = list;
    size_ = coefs_.size();
}
PolynomMatrix::PolynomMatrix(const PolynomMatrix& p) {
    coefs_ = p.coefs_;
    size_ = p.size_;
}
PolynomMatrix::PolynomMatrix(PolynomMatrix&& p) {
    coefs_ = std::move(p.coefs_);
    size_ = p.size_;
    p.size_ = 0;
}
PolynomMatrix& PolynomMatrix::operator=(const PolynomMatrix& other) {
    coefs_ = other.coefs_;
    size_ = other.size_;
    return *this;
}
PolynomMatrix& PolynomMatrix::operator=(PolynomMatrix&& other) {
    coefs_ = std::move(other.coefs_);
    size_ = other.size_;
    other.size_ = 0;
    return *this;
}
Matrix& PolynomMatrix::operator[](int idx) {
    return coefs_[idx];
}
const Matrix& PolynomMatrix::operator[](int idx) const {
    return coefs_[idx];
}
PolynomMatrix& PolynomMatrix::operator*(double val) {
    for (auto& element : coefs_) {
        element *= val;
    }
    return *this;
}
PolynomMatrix& PolynomMatrix::operator*(const PolynomMatrix& p) {
    throw std::logic_error("not implemented!");
}
Matrix PolynomMatrix::GetValue(double val) {
    Matrix ans(coefs_[0].rows_, coefs_[0].rows_, 0);
    Matrix tmp = ans;
    if (std::abs(val) <= 1) {
        for (int idx = size_ - 1; idx >= 0; --idx) {
            tmp = coefs_[idx] * ComputePower(val, idx);
            ans = ans + tmp;
        }
        return ans;
    }
    int helper = std::floor(val);
    val /= helper;
    double power = 1;
    for (int idx = 0; idx < size_; idx++) {
        tmp = coefs_[idx] * power;
        power *= val;
        ans = ans + tmp;
    }
    ans = ans ^ helper;
    return ans;
}
std::ostream& operator<<(std::ostream& os, const PolynomMatrix& poly) {
    for (int idx = poly.size_ - 1; idx >= 0; idx--) {
        std::string suffix;
        os << poly[idx];
        if (idx != 0) {
            suffix += "x^" + std::to_string(idx);
        }
        os << suffix << '\n';
    }
    return os;
}
Matrix operator*(double val, const Matrix& matrix) {
    return matrix * val;
}