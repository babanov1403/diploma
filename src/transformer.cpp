#include "transformer.h"
Transformer::Transformer() {
}
Matrix Transformer::ComputeX0(double timepoint) {
    Matrix matrix_x = exp_row_a_.GetValue(timepoint);
    matrix_x = matrix_x * exp_first_time_point_inv_a_;
    matrix_x = matrix_x * matrix_x0_;
    return matrix_x;
}
Matrix Transformer::ComputeG0() {
    return matrix_g_ + matrix_h_ * ComputeX0(second_time_point_) * (-1);
}
Matrix Transformer::ComputeG(double timepoint) {
    return matrix_h_ * exp_second_time_point_a_ * (exp_row_inv_a_.GetValue(timepoint));
}

Matrix Transformer::ComputeC(double left_t, double right_t, size_t interval_cnt) {
    Matrix matrix_c(1, interval_cnt);
    double step = (right_t - left_t) / interval_cnt;
    int k = 0;
    for (double idx = left_t; idx <= right_t - step; idx += step, ++k) {
        matrix_c(0, k) = GetCh(idx, idx + step);
    }
    return matrix_c;
}

Matrix Transformer::ComputeD(double left_t, double right_t, size_t interval_cnt) {
    double step = (right_t - left_t) / interval_cnt;
    Matrix matrix_d(matrix_g_.GetRows(), interval_cnt, 0);
    #pragma omp parallel for
    for (int quant = 0; quant <= number_; quant++) {
        double idx = left_t + quant * step;
        Matrix curr = GetDh(idx, idx + step, right_t);

        for (int jdx = 0; jdx < matrix_d.GetRows(); jdx++) {
            matrix_d(jdx, quant) = curr(jdx, 0);
        }
    }
    return matrix_d;
}
// x(Tk+1) = Y(Tk+1)*Y^-1(Xk)*x0 + Y(Tk+1) * integrale(xk -> xk+1, Y^-1(t)*b dt)
Matrix Transformer::ComputePathWithGivenU(const std::vector<double>& u_vec,
                                          std::filesystem::path path) {
    double step = (second_time_point_ - first_time_point_) / number_;
    Matrix curr = matrix_x0_;
    int k = 0;
    if (path.empty()) {
        for (double idx = step; idx <= second_time_point_; idx += step, ++k) {
            curr = exp_row_a_.GetValue(idx) * (exp_row_inv_a_.GetValue(idx - step) * curr +
                                               ComputeIntegraleCauchy(idx - step, idx, u_vec[k]));
        }
    } else {
        int dim = curr.GetRows();
        std::ofstream fout(path);
        bool flag = true;
        fout << "{";
        constexpr double kPrecision = 1e2;
        for (double idx = step; idx <= second_time_point_; idx += step, k++) {
            for (double jdx = idx - step + step / kPrecision; jdx <= idx;
                 jdx += step / kPrecision) {
                Matrix temporary =
                    exp_row_a_.GetValue(jdx) * (exp_row_inv_a_.GetValue(idx - step) * curr +
                                                ComputeIntegraleCauchy(idx - step, jdx, u_vec[k]));
                fout << "'" << jdx << "':[";
                for (int inner_idx = 0; inner_idx < dim; inner_idx++) {
                    if (flag) {
                        fout << matrix_x0_(inner_idx, 0);
                    } else {
                        fout << temporary(inner_idx, 0);
                    }
                    if (inner_idx != dim - 1) {
                        fout << ',';
                    }
                }
                fout << "],";
                flag = false;
            }
            curr = exp_row_a_.GetValue(idx) * (exp_row_inv_a_.GetValue(idx - step) * curr +
                                               ComputeIntegraleCauchy(idx - step, idx, u_vec[k]));
        }
        fout << "}";
        fout.close();
    }
    return curr;
}
// hardcode all input params right here
void Transformer::Initialize() {
    double k1 = 1, k2 = k1, k3 = k1;
    double m1 = 1, m2 = 1, m3 = 1;
    double mu = 0;
    matrix_a_ = {{0, 0, 0, 1, 0, 0},
                 {0, 0, 0, 0, 1, 0},
                 {0, 0, 0, 0, 0, 1},
                 {-1.0 * (k1 + k2) / m1, k2 / m1, 0, mu, 0, 0},
                 {k2 / m2, (-k3 - k2) / m2, k3 / m2, 0, mu, 0},
                 {0, k3 / m3, -k3 / m3, 0, 0, mu}};
    exp_row_a_ = GetExponentPolynom(matrix_a_);
    exp_row_inv_a_ = GetExponentPolynom(matrix_a_ * (-1));
    matrix_b_ = Matrix{0, 0, 0, 0, 0, 1.0 / m3}.GetTransposed();
    matrix_x0_ = Matrix{1, 1, 1, 0, 0, 0}.GetTransposed();
    first_time_point_ = 0;
    second_time_point_ = 80;
    exp_first_time_point_a_ = exp_row_a_.GetValue(first_time_point_);
    exp_second_time_point_a_ = exp_row_a_.GetValue(second_time_point_);
    exp_first_time_point_inv_a_ = exp_row_inv_a_.GetValue(first_time_point_);
    exp_second_time_point_inv_a_ = exp_row_inv_a_.GetValue(second_time_point_);
    number_ = 1'000'000;
    matrix_c_ = Matrix{1, 1, 1, 1, 1, 1}.GetTransposed();
    matrix_h_ = Matrix(6);
    matrix_g_ = Matrix{0, 0, 0, 0, 0, 0}.GetTransposed();

    // hardcoded

    matrix_c_u_ = Matrix(number_, 1, 1);
}
double Transformer::GetCh(double left_t, double right_t) {
    auto nodes = MakeNodes(kNodesCount, left_t, right_t);
    Matrix ajth = MakeISFCD(nodes);
    int size = ajth.GetRows();
    double result_integrale = 0;
    std::vector<double> parts(size);
    auto func_ch = [this](double val) {
        Matrix result = matrix_c_.GetTransposed();
        result *= exp_second_time_point_a_;
        result *= exp_row_inv_a_.GetValue(val);
        result *= matrix_b_;
        return result[0][0];
    };
    for (int idx = 0; idx < size; idx++) {
        parts[idx] = ajth(idx, 0) * func_ch(nodes[idx]);
    }
    std::ranges::sort(parts);
    for (auto el : parts) {
        result_integrale += el;
    }
    return result_integrale;
}
static bool zhopa = true;
Matrix Transformer::GetDh(double left_t, double right_t, double end_t) {
    auto nodes = MakeNodes(kNodesCount, left_t, right_t);
    Matrix ajth = MakeISFCD(nodes);
    int size = ajth.GetRows();
    auto func_dh = [this](double val) { return ComputeG(val) * matrix_b_; };
    Matrix integrale = func_dh(nodes[0]) * ajth(0, 0);
    for (int idx = 1; idx < size; idx++) {
        integrale += func_dh(nodes[idx]) * ajth(idx, 0);
    }
    return integrale;
}
Matrix Transformer::FuncCauchy(double val, const Matrix& matrix_b, double u_val) {
    return exp_row_inv_a_.GetValue(val) * matrix_b * u_val;
}
Matrix Transformer::MakeISFCD(const std::vector<double>& nodes) {
    Matrix moments(nodes.size(), 1, 0);
    double left = nodes[0];
    double right = nodes.back();
    for (int idx = 0; idx < nodes.size(); idx++) {
        moments(idx, 0) = ComputeMoment(idx, left, right);
    }
    Matrix matrix_a(nodes.size(), nodes.size());
    for (int idx = 0; idx < nodes.size(); idx++) {
        for (int jdx = 0; jdx < nodes.size(); jdx++) {
            matrix_a(idx, jdx) = ComputePower(nodes[jdx], idx);
        }
    }
    Matrix output = Matrix::SolveLinearSystem(matrix_a, moments);
    return output;
}
Matrix Transformer::ComputeIntegraleCauchy(double left, double right, double u_val) {
    std::vector<double> nodes = MakeNodes(kNodesCount, left, right);
    Matrix ajth = MakeISFCD(nodes);
    int size = ajth.GetRows();
    Matrix integrale = FuncCauchy(nodes[0], matrix_b_, u_val) * ajth(0, 0);
    for (int idx = 1; idx < size; idx++) {
        integrale += FuncCauchy(nodes[idx], matrix_b_, u_val) * ajth(idx, 0);
    }
    return integrale;
}
size_t Transformer::GetNumber() const {
    return number_;
}
Matrix Transformer::GetC() const {
    return matrix_c_;
}
double Transformer::GetFirstTimepoint() const {
    return first_time_point_;
}
double Transformer::GetSecondTimepoint() const {
    return second_time_point_;
}
Matrix Transformer::GetH() const {
    return matrix_h_;
}
Matrix Transformer::Getg() const {
    return matrix_g_;
}