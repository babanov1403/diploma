#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <ranges>


#include "integration_utils.h"
#include "matrix.h"
#include "polynom.h"
#include "utils.h"
/*
    takes input matrices, computes output matrices for
    linear programming task
*/
class Transformer {
public:
    Transformer();

    Matrix ComputeX0(double timepoint);

    Matrix ComputeG0();

    Matrix ComputeG(double timepoint);
    // TODO: fix semantics
    Matrix ComputeC(double left_t, double right_t, size_t interval_cnt);

    Matrix ComputeD(double left_t, double right_t, size_t interval_cnt);

    // x(Tk+1) = Y(Tk+1)*Y^-1(Xk)*x0 + Y(Tk+1) * integrale(xk -> xk+1, Y^-1(t)*b dt)
    Matrix ComputePathWithGivenU(const std::vector<double>& u_vec,
                                 std::filesystem::path path = "");

    // hardcode all input params right here
    void Initialize();
    size_t GetNumber() const;
    Matrix GetC() const;
    double GetFirstTimepoint() const;
    double GetSecondTimepoint() const;
    Matrix GetH() const;
    Matrix Getg() const;

    friend class Solver;

private:
    double GetCh(double left_t, double right_t);

    Matrix GetDh(double left_t, double right_t, double end_t);

    Matrix FuncCauchy(double val, const Matrix& matrix_b, double u_val);
    Matrix MakeISFCD(const std::vector<double>& nodes);

    Matrix ComputeIntegraleCauchy(double left, double right, double u_val);

    Matrix matrix_a_;
    Matrix matrix_c_;
    Matrix matrix_h_;
    Matrix matrix_g_;
    Matrix matrix_b_;
    Matrix matrix_x0_;
    PolynomMatrix exp_row_a_;
    PolynomMatrix exp_row_inv_a_;
    double first_time_point_;
    double second_time_point_;
    Matrix exp_first_time_point_a_;
    Matrix exp_second_time_point_a_;
    Matrix exp_first_time_point_inv_a_;
    Matrix exp_second_time_point_inv_a_;
    size_t number_;
};