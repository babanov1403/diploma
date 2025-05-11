#include "solver.h"

Solver::Solver() {
}

void Solver::SolveRegular(alglib::minlpstate &state, alglib::minlpreport &rep, alglib::real_1d_array &x, Transformer* transformer) {
    alglib::real_2d_array d_1;
    size_t number = transformer->number_;
    d_1.setlength(transformer->matrix_h_.GetRows(), number);
    alglib::real_1d_array target;
    target.setlength(number);
    alglib::real_1d_array l_bound_u;
    l_bound_u.setlength(number);
    alglib::real_1d_array u_bound_u;
    u_bound_u.setlength(number);
    alglib::real_1d_array l_bound_d;
    l_bound_d.setlength(transformer->matrix_h_.GetRows());
    alglib::real_1d_array u_bound_d;
    u_bound_d.setlength(transformer->matrix_h_.GetRows());
    alglib::real_1d_array scale;
    scale.setlength(number);

    // cout << D << '\n';
    // cout << C << '\n';

    Matrix matrix_d = transformer->ComputeD(transformer->first_time_point_, transformer->second_time_point_, number);
    // std::cout << matrix_d.GetRows() << ' ' << matrix_d.GetColumns() << '\n';
    alglib::minlpcreate(number, state);
    for (int idx = 0; idx < matrix_d.GetRows(); idx++) {
        for (int jdx = 0; jdx < matrix_d.GetColumns(); jdx++) {
            d_1(idx, jdx) = matrix_d(idx, jdx);
        }
    }

    for (int idx = 0; idx < number; idx++) {
        target(idx) = -transformer->matrix_c_(0, idx);
    }
    
    for (int idx = 0; idx < number; idx++) {
        l_bound_u(idx) = 0;
        u_bound_u(idx) = 10;
    }

    Matrix matrix_g0 = transformer->ComputeG0();

    for (int idx = 0; idx < transformer->matrix_h_.GetRows(); idx++) {
        l_bound_d(idx) = matrix_g0(idx, 0);
        u_bound_d(idx) = l_bound_d(idx);
    }

    for (int idx = 0; idx < number; idx++) {
        scale(idx) = 1;
    }

    alglib::minlpsetcost(state, target);
    alglib::minlpsetbc(state, l_bound_u, u_bound_u);
    alglib::minlpsetlc2dense(state, d_1, l_bound_d, u_bound_d);
    alglib::minlpsetscale(state, scale);

    alglib::minlpoptimize(state);
    alglib::minlpresults(state, x, rep);

    if (rep.terminationtype <= 4 && rep.terminationtype >= 1) {
        std::cout << "\nLP OK\n";
    } else {
        std::cout << "\nLP NOT OK\n";
    }
}