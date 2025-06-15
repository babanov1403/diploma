#pragma once
#include <ranges>
#include <set>
#include <unordered_set>

#include "alglib/optimization.h"
#include "transformer.h"

/*

1) write it on python with numpy, and when it will work - rewrite to cpp
2) update matrix library, and write this algo



*/

constexpr double kSolverTolerance = 0.5;

class Solver {
public:
    Solver();
    Solver(Transformer* transformer);
    void SolveRegular(alglib::minlpstate& state, alglib::minlpreport& rep,
                      alglib::real_1d_array& x, const Matrix& matrix_d, const Matrix& matrix_g0);
    Matrix SolveAdaptiveRaw(const Matrix& matrix_c, const Matrix& matrix_a, const Matrix& matrix_lb,
                            const Matrix& matrix_ub, const Matrix& matrix_ld,
                            const Matrix& matrix_ud);
    Matrix SolveAdaptive();

private:
    Transformer* transformer_;
};