#include "solver.h"
Solver::Solver() {
}
Solver::Solver(Transformer* transformer) : transformer_(transformer) {
    // build here
    // matrix_c, matrix_a, matrix_lb, matrix_ub, matrix_ld, matrix_ud
    // matrix_c == matrix_c_
    // matrix_a == matrix_d_
    // matrix_lb == matrix_g0_
    // matrix_ub == matrix_g0_
    // matrix_ld and matrix_ud manually
}
void Solver::SolveRegular(alglib::minlpstate& state, alglib::minlpreport& rep,
                          alglib::real_1d_array& x, const Matrix& matrix_d, const Matrix& matrix_g0) {
    int xxx = 0;
    while (xxx++ < 1) {
        alglib::real_2d_array d_1;
    size_t number = transformer_->number_;
    d_1.setlength(transformer_->matrix_h_.GetRows(), number);
    alglib::real_1d_array target;
    target.setlength(number);
    alglib::real_1d_array l_bound_u;
    l_bound_u.setlength(number);
    alglib::real_1d_array u_bound_u;
    u_bound_u.setlength(number);
    alglib::real_1d_array l_bound_d;
    l_bound_d.setlength(transformer_->matrix_h_.GetRows());
    alglib::real_1d_array u_bound_d;
    u_bound_d.setlength(transformer_->matrix_h_.GetRows());
    alglib::real_1d_array scale;
    scale.setlength(number);
    // cout << D << '\n';
    // cout << C << '\n';
    // std::cout << matrix_d.GetRows() << ' ' << matrix_d.GetColumns() << '\n';
    alglib::minlpcreate(number, state);
    for (int idx = 0; idx < matrix_d.GetRows(); idx++) {
        for (int jdx = 0; jdx < matrix_d.GetColumns(); jdx++) {
            d_1(idx, jdx) = matrix_d(idx, jdx);
        }
    }
    for (int idx = 0; idx < number; idx++) {
        target(idx) = -transformer_->matrix_c_u_(idx, 0);
    }
    for (int idx = 0; idx < number; idx++) {
        l_bound_u(idx) = 0;
        u_bound_u(idx) = 10;
    }
    for (int idx = 0; idx < transformer_->matrix_h_.GetRows(); idx++) {
        l_bound_d(idx) = matrix_g0(idx, 0) - 0.1;
        u_bound_d(idx) = l_bound_d(idx) + 0.1;
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
        // std::cout << "\nLP OK\n";
    } else {
        std::cout << "\nLP NOT OK\n";
    }
    }
}
Matrix Solver::SolveAdaptiveRaw(const Matrix& matrix_c, const Matrix& matrix_a,
                                const Matrix& matrix_lb, const Matrix& matrix_ub,
                                const Matrix& matrix_ld, const Matrix& matrix_ud) {
    // STEP 0: CHOOSING OPORNIY PLAN
    auto x = (matrix_ld + matrix_ud) / 2;
    // x = x.GetTransposed();
    std::vector<int> basis_i;  // from 0 to matrix_a.GetRows() - 1
    std::vector<int> basis_j;  // from 0 to matrix_a.GetColumns() - 1
    std::vector<int> basis_i_residual;
    for (int idx = 0; idx < matrix_a.GetRows(); idx++) {
        basis_i_residual.emplace_back(idx);
    }
    std::vector<int> basis_j_residual;
    for (int idx = 0; idx < matrix_a.GetColumns(); idx++) {
        basis_j_residual.emplace_back(idx);
    }
    int iter = 0;
    while (true) {
        std::cout << "============================\n";
        std::cout << "RUNNING ON ITERATION: " << iter++ << '\n';
        std::ranges::sort(basis_i);
        std::ranges::sort(basis_j);
        std::ranges::sort(basis_i_residual);
        std::ranges::sort(basis_j_residual);
        int opora_dim = basis_i.size();
        Matrix a_op;
        Matrix a_op_inv;
        if (opora_dim > 0) {
            a_op = matrix_a.GetSubMatrix(basis_i, basis_j);
            a_op_inv = a_op.GetInversed();
        }
        auto omega_lower = matrix_lb - matrix_a * x;
        auto omega_upper = matrix_ub - matrix_a * x;
        // STEP 1: compute u && delta
        Matrix u(matrix_a.GetRows(), 1, 0);
        Matrix delta(matrix_a.GetColumns(), 1, 0);
        if (opora_dim > 0) {
            u.UpdateMatrix(a_op_inv.GetTransposed() * matrix_c.GetSubMatrixByRow(basis_j), basis_i,
                           {0});
            auto a_iop_jn = matrix_a.GetSubMatrixByRow(basis_i).GetTransposed();
            delta = a_iop_jn * u.GetSubMatrixByRow(basis_i) - matrix_c;
        } else {
            delta = (-1) * matrix_c;
        }
        // STEP 2: optimum criterion
        auto z = matrix_a * x;
        bool is_optimum = true;
        for (auto idx : basis_i) {
            if (u(idx, 0) <= 0 && EqualNotStrict(z(idx, 0), matrix_lb(idx, 0))) {
                continue;
            }
            if (u(idx, 0) >= 0 && EqualNotStrict(z(idx, 0), matrix_ub(idx, 0))) {
                continue;
            }
            if (EqualNotStrict(u(idx, 0), 0) && z(idx, 0) < matrix_ub(idx, 0) &&
                z(idx, 0) > matrix_lb(idx, 0)) {
                continue;
            }
            is_optimum = false;
        }
        for (auto jdx : basis_j_residual) {
            if (delta(jdx, 0) <= 0 && EqualNotStrict(x(jdx, 0), matrix_ud(jdx, 0))) {
                continue;
            }
            if (delta(jdx, 0) >= 0 && EqualNotStrict(x(jdx, 0), matrix_ld(jdx, 0))) {
                continue;
            }
            if (EqualNotStrict(delta(jdx, 0), 0) && x(jdx, 0) < matrix_ud(jdx, 0) &&
                x(jdx, 0) > matrix_ld(jdx, 0)) {
                continue;
            }
            is_optimum = false;
        }
        if (is_optimum) {
            return x;
        }
        // STEP 3: calculate beta
        // maybe some problems here, check with variety of epsilon
        double beta = 0;
        for (auto jdx = 0; jdx < delta.GetRows(); jdx++) {
            if (delta(jdx, 0) > 0) {
                beta += delta(jdx, 0) * (x(jdx, 0) - matrix_ld(jdx, 0));
            } else {
                beta += delta(jdx, 0) * (x(jdx, 0) - matrix_ud(jdx, 0));
            }
        }
        for (auto idx : basis_i) {
            if (u(idx, 0) < 0) {
                beta += u(idx, 0) * omega_lower(idx, 0);
            } else {
                beta += u(idx, 0) * omega_upper(idx, 0);
            }
        }
        if (beta <= kSolverTolerance) {
            return x;
        }
        Matrix l(x.GetRows(), 1, 0);
        for (auto jdx : basis_j_residual) {
            if (delta(jdx, 0) < 0) {
                l(jdx, 0) = matrix_ud(jdx, 0) - x(jdx, 0);
            } else if (delta(jdx, 0) > 0) {
                l(jdx, 0) = matrix_ld(jdx, 0) - x(jdx, 0);
            } else {
                l(jdx, 0) = 0;
            }
        }
        Matrix omega(matrix_a.GetRows(), 1, 0);
        for (auto idx : basis_i) {
            if (u(idx, 0) < 0) {
                omega(idx, 0) = omega_lower(idx, 0);
            } else if (u(idx, 0) > 0) {
                omega(idx, 0) = omega_upper(idx, 0);
            } else {
                omega(idx, 0) = 0;
            }
        }
        if (opora_dim > 0) {
            l.UpdateMatrix(a_op_inv * omega.GetSubMatrixByRow(basis_j) -
                               a_op_inv * matrix_a.GetSubMatrix(basis_i, basis_j_residual) *
                                   l.GetSubMatrixByRow(basis_j_residual),
                           basis_j, {0});
        }
        // STEP 5: computing theta
        double theta = 1;
        int bad_index = -1;
        int bad_index_type = -1;  // 0 - from basis_j, 1 - from basis_i_residual
        double theta_j = 1e9;
        int theta_j_idx = -1;
        double theta_i = 1e9;
        int theta_i_idx = -1;
        for (auto jdx : basis_j) {
            if (l(jdx, 0) < 0) {
                double val = (matrix_ld(jdx, 0) - x(jdx, 0)) / l(jdx, 0);
                if (val < theta_j) {
                    theta_j = val;
                    theta_j_idx = jdx;
                }
            } else if (l(jdx, 0) > 0) {
                double val = (matrix_ud(jdx, 0) - x(jdx, 0)) / l(jdx, 0);
                if (val < theta_j) {
                    theta_j = val;
                    theta_j_idx = jdx;
                }
            }
        }
        std::cout << "l:\n" << l;
        std::cout << "x:\n" << x;
        for (auto idx : basis_i) {
            std::cout << idx << ' ';
        }
        std::cout << '\n';
        for (auto idx : basis_i_residual) {
            std::cout << idx << ' ';
        }
        std::cout << '\n';
        for (auto idx : basis_j) {
            std::cout << idx << ' ';
        }
        std::cout << '\n';
        for (auto idx : basis_j_residual) {
            std::cout << idx << ' ';
        }
        std::cout << '\n';
        std::cout << "omegas:\n";
        std::cout << omega_upper;
        std::cout << omega_lower;
        auto rows_dir = matrix_a * l;
        std::cout << "rows dir:\n";
        std::cout << rows_dir;
        for (auto idx : basis_i_residual) {
            if (rows_dir(idx, 0) < 0) {
                double val = omega_lower(idx, 0) / rows_dir(idx, 0);
                if (val < theta_i) {
                    theta_i = val;
                    theta_i_idx = idx;
                }
            } else if (rows_dir(idx, 0) > 0) {
                double val = omega_upper(idx, 0) / rows_dir(idx, 0);
                if (val < theta_i) {
                    theta_i = val;
                    theta_i_idx = idx;
                }
            }
        }
        if (theta > theta_i) {
            theta = theta_i;
            bad_index = theta_i_idx;
            bad_index_type = 1;
        }
        if (theta > theta_j) {
            theta = theta_j;
            bad_index = theta_j_idx;
            bad_index_type = 0;
        }
        // STEP 6: compute x_bar and beta_bar
        std::cout << "theta: " << theta << '\n';
        std::cout << theta_i << " " << theta_j << '\n';
        x = x + l * theta;
        double beta_bar = (1 - theta) * beta;
        if (beta_bar <= kSolverTolerance) {
            return x;
        }
        Matrix ksi_j(matrix_a.GetColumns(), 1, 0);
        Matrix ksi_i(matrix_a.GetRows(), 1, 0);
        Matrix matrix_j_res(matrix_a.GetRows(), matrix_a.GetColumns(), 0);
        Matrix matrix_i(matrix_a.GetRows(), matrix_a.GetColumns(), 0);
        double alpha = 0;
        std::cout << "bad index type: " << bad_index << ' ' << bad_index_type << '\n';
        if (bad_index_type == 0) {
            int sign = x(bad_index, 0) == matrix_ld(bad_index, 0) ? 1 : -1;
            if (opora_dim > 0) {
                matrix_j_res.UpdateMatrix(
                    a_op_inv * matrix_a.GetSubMatrix(basis_i, basis_j_residual), basis_i,
                    basis_j_residual);
                matrix_i.UpdateMatrix(-1 * a_op_inv, basis_i, basis_j);
                ksi_j.UpdateMatrix(
                    matrix_j_res.GetSubMatrixByRow({bad_index}).GetTransposed() * sign,
                    basis_j_residual, {0});
                ksi_i.UpdateMatrix(matrix_i.GetSubMatrixByRow({bad_index}).GetTransposed(), basis_i,
                                   {0});
            }
            if (sign > 0) {
                alpha = x(bad_index, 0) + l(bad_index, 0) - matrix_ld(bad_index, 0);
            } else {
                alpha = matrix_ud(bad_index, 0) - x(bad_index, 0) - l(bad_index, 0);
            }
        } else {
            int sign = ((matrix_a.GetSubMatrixByRow({bad_index}) * x) ==
                        matrix_ub.GetSubMatrixByRow({bad_index}))
                           ? 1
                           : -1;
            std::cout << matrix_a.GetSubMatrixByRow({bad_index}) * x;
            std::cout << matrix_ub.GetSubMatrixByRow({bad_index});
            if (opora_dim > 0) {
                matrix_j_res.UpdateMatrix(
                    matrix_a.GetSubMatrix(basis_i_residual, basis_j_residual) -
                        matrix_a.GetSubMatrix(basis_i_residual, basis_j) * a_op_inv *
                            matrix_a.GetSubMatrix(basis_i, basis_j_residual),
                    basis_i_residual, basis_j_residual);
                matrix_i.UpdateMatrix(matrix_a.GetSubMatrix(basis_i_residual, basis_j) * a_op_inv,
                                      basis_i_residual, basis_j);
                ksi_j.UpdateMatrix(matrix_j_res.GetSubMatrixByRow({bad_index})
                                           .GetTransposed()
                                           .GetSubMatrixByRow(basis_j_residual) *
                                       sign,
                                   basis_j_residual, {0});
                ksi_i.UpdateMatrix(matrix_i.GetSubMatrixByRow({bad_index})
                                           .GetTransposed()
                                           .GetSubMatrixByRow(basis_j) *
                                       sign,
                                   basis_i, {0});
            } else {
                matrix_j_res = matrix_a.GetSubMatrix(basis_i_residual, basis_j_residual);
                std::cout << matrix_j_res.GetSubMatrixByRow({bad_index});
                ksi_j.UpdateMatrix(
                    matrix_j_res.GetSubMatrixByRow({bad_index}).GetTransposed() * sign,
                    basis_j_residual, {0});
            }
            std::cout << matrix_j_res;
            std::cout << ksi_j;
            auto tmp = matrix_a.GetSubMatrixByRow({bad_index}) * (x + l);
            if (sign > 0) {
                alpha = matrix_ub(bad_index, 0) - tmp(0, 0);
            } else {
                alpha = tmp(0, 0) - matrix_lb(bad_index, 0);
            }
        }
        std::cout << "alpha right after: " << alpha << '\n';
        // setting up kappa, delta...
        Matrix delta_j(matrix_a.GetColumns(), 1, 0);
        Matrix delta_i(matrix_a.GetRows(), 1, 0);
        delta_j.UpdateMatrix(delta.GetSubMatrixByRow(basis_j_residual), basis_j_residual, {0});
        if (opora_dim > 0) {
            delta_i.UpdateMatrix(u.GetSubMatrixByRow(basis_i) * (-1), basis_i, {0});
        }
        Matrix kappa_j(matrix_a.GetColumns(), 1, 0);
        Matrix kappa_i(matrix_a.GetRows(), 1, 0);
        kappa_j.UpdateMatrix(
            x.GetSubMatrixByRow(basis_j_residual) + l.GetSubMatrixByRow(basis_j_residual),
            basis_j_residual, {0});
        if (opora_dim > 0) {
            kappa_i.UpdateMatrix((matrix_a * (x + l)).GetSubMatrixByRow(basis_i), basis_i, {0});
        }
        double sigma = 0;
        int bad_index_2 = -1;
        int bad_index_2_type = -1;
        std::cout << "ksi_j:\n";
        std::cout << ksi_j;
        std::cout << "ksi_i:\n";
        std::cout << ksi_i;
        std::cout << "kappa_j:\n";
        std::cout << kappa_j;
        std::cout << "kappa_i:\n";
        std::cout << kappa_i;
        while (314 > 313) {
            double sigma_tmp = 1e9;
            for (int idx = 0; idx < ksi_j.GetRows(); idx++) {
                if (ksi_j(idx, 0) * delta_j(idx, 0) < 0) {
                    double val = -delta_j(idx, 0) / ksi_j(idx, 0);
                    if (val < sigma_tmp) {
                        sigma_tmp = val;
                        bad_index_2 = idx;
                        bad_index_2_type = 0;
                    }
                }
            }
            for (int idx = 0; idx < ksi_i.GetRows(); idx++) {
                if (ksi_i(idx, 0) * delta_i(idx, 0) < 0) {
                    double val = -delta_i(idx, 0) / ksi_i(idx, 0);
                    if (val < sigma_tmp) {
                        sigma_tmp = val;
                        bad_index_2 = idx;
                        bad_index_2_type = 1;
                    }
                }
            }
            if (sigma_tmp > 1e8) {
                break;
            }
            for (int idx = 0; idx < ksi_j.GetRows(); idx++) {
                delta_j(idx, 0) += sigma_tmp * ksi_j(idx, 0);
                if (EqualNotStrict(delta_j(idx, 0), 0)) {
                    if (ksi_j(idx, 0) < 0) {
                        alpha += ksi_j(idx, 0) * (kappa_j(idx, 0) - matrix_ud(idx, 0));
                    } else {
                        alpha += ksi_j(idx, 0) * (kappa_j(idx, 0) - matrix_ld(idx, 0));
                    }
                }
            }
            for (int idx = 0; idx < ksi_i.GetRows(); idx++) {
                delta_i(idx, 0) += sigma_tmp * ksi_i(idx, 0);
                if (EqualNotStrict(delta_i(idx, 0), 0)) {
                    if (ksi_i(idx, 0) < 0) {
                        alpha += ksi_i(idx, 0) * (kappa_i(idx, 0) - matrix_ub(idx, 0));
                    } else {
                        alpha += ksi_i(idx, 0) * (kappa_i(idx, 0) - matrix_lb(idx, 0));
                    }
                }
            }
            sigma += sigma_tmp;
            std::cout << "alpha: " << alpha << '\n';
            if (alpha >= 0) {
                break;
            }
        }
        // STEP 9: four variants!
        if (bad_index_2 == -1) {
            // throw std::runtime_error("bad_index_2 == -1!");
            return x;
        }
        if (bad_index_type == 1 && bad_index_2_type == 1) {
            basis_i.emplace_back(bad_index);
            std::erase(basis_i, bad_index_2);
            basis_i_residual.emplace_back(bad_index_2);
            std::erase(basis_i_residual, bad_index);
        } else if (bad_index_type == 1 && bad_index_2_type == 0) {
            basis_i.emplace_back(bad_index);
            basis_j.emplace_back(bad_index_2);
            std::erase(basis_i_residual, bad_index);
            std::erase(basis_j_residual, bad_index_2);
        } else if (bad_index_type == 0 && bad_index_2_type == 1) {
            std::erase(basis_i, bad_index_2);
            std::erase(basis_j, bad_index);
            basis_i_residual.emplace_back(bad_index_2);
            basis_j_residual.emplace_back(bad_index);
        } else {
            basis_j.emplace_back(bad_index_2);
            std::erase(basis_j, bad_index);
            std::erase(basis_j_residual, bad_index_2);
            basis_j_residual.emplace_back(bad_index);
        }
    }
}
Matrix Solver::SolveAdaptive() {
    auto number = transformer_->number_;
    Matrix matrix_c = transformer_->matrix_c_u_;
    Matrix matrix_a = transformer_->ComputeD(transformer_->first_time_point_,
                                             transformer_->second_time_point_, number);
    Matrix matrix_lb = transformer_->ComputeG0();
    Matrix matrix_ub = transformer_->ComputeG0();
    Matrix matrix_ld(matrix_c.GetRows(), 1, 0);
    Matrix matrix_ud(matrix_c.GetRows(), 1, 0);
    for (int idx = 0; idx < matrix_c.GetRows(); idx++) {
        matrix_ld(idx, 0) = 0;
        matrix_ud(idx, 0) = 10;
    }

    std::cout << "matrix_c:\n";
    std::cout << matrix_c;
    std::cout << "matrix_a:\n";
    std::cout << matrix_a;
    std::cout << "matrix_lb:\n";
    std::cout << matrix_lb;
    std::cout << "matrix_ub:\n";
    std::cout << matrix_ub;
    std::cout << "matrix_ld:\n";
    std::cout << matrix_ld;
    std::cout << "matrix_ud:\n";
    std::cout << matrix_ud;
    return {};


    return SolveAdaptiveRaw(matrix_c, matrix_a, matrix_lb, matrix_ub, matrix_ld, matrix_ud);
}