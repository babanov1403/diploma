#pragma once
#include "alglib/optimization.h"
#include "transformer.h"

#include <unordered_set>
#include <set>
#include <ranges>

/*

1) write it on python with numpy, and when it will work - rewrite to cpp
2) update matrix library, and write this algo



*/

constexpr double kSolverTolerance = 0.5;

template <class T> 
int FindPosOfIndex(const std::set<T>& set, int val) {
    std::vector vec(set.begin(), set.end());
    for (int idx = 0; idx < vec.size(); idx++) {
        if (vec[idx] == val) {
            return idx;
        }
    }
    return -1;
}


class Solver {
    enum class INDEX_TYPE { BOUND, VARIABLE };
    enum class INDEX_TYPE_PLAN { BASIS_J, BASIS_I_INV };
    enum class INDEX_TYPE_PLAN_2 { BASIS_J_INV, BASIS_I };
public:
    Solver();
    void SolveRegular(alglib::minlpstate &state, alglib::minlpreport &rep, alglib::real_1d_array &x, Transformer* transformer);
    auto SolveAdaptive(const Matrix& matrix_c, const Matrix& matrix_a, const Matrix& matrix_lb, const Matrix& matrix_ub
                     , const Matrix& matrix_ld, const Matrix& matrix_ud) {
        /*
        matrix_c = Nx1
        matrix_a = MxN
        matrix_lb, matrix_ub = Mx1
        matrix_ld, matrix_ud = Nx1
        */

        // TODO: maybe transposing is tricky to, and we need
        // first get submatrix, and only then transpose?
        Matrix matrix_a_transposed = matrix_a.GetTransposed();
        // STEP 0: CHOOSING OPORNIY PLAN
        auto x = (matrix_ld + matrix_ud) / 2;
        // TODO: rofls
        x = Matrix{{2, 1}};
        x = x.GetTransposed();
        std::vector<int> basis_i; // from 0 to matrix_a.GetRows() - 1
        std::vector<int> basis_j; // from 0 to matrix_a.GetColumns() - 1

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
            std::cout << iter++ << '\n';
            // STEP 1: compute (u) and (\delta) vector

            auto calculate_potential_delta = [&]{
                Matrix u(matrix_a.GetRows(), 1, 0);
                Matrix delta(matrix_a.GetColumns(), 1, 0);
                Matrix res;
                if (!basis_i.empty() && !basis_j.empty()) {
                    auto matrix_a_inversed = matrix_a.GetSubMatrix(basis_i, basis_j).GetInversed();
                    auto matrix_c_submatrix = matrix_c.GetSubMatrix(basis_i, {0});
                    res = matrix_c_submatrix.GetTransposed() * matrix_a_inversed;
                    int idx_helper = 0;
                    for (auto idx : basis_i) {
                        u(idx, 0) = res(0, idx_helper++);
                    }   
                }

                for (int jdx = 0; jdx < delta.GetRows(); jdx++) {
                    for (auto idx : basis_i) {
                        delta(jdx, 0) -= u(idx, 0) * matrix_a(idx, jdx);
                    }
                }

                delta += matrix_c;

                return std::make_pair(u, delta);
            };

            auto [u, delta] = calculate_potential_delta();
            std::cout << "u and delta:\n";
            std::cout << u << delta;

            // STEP 2: checking criterio, if return array is empty, then stop the execution
            // otherwise, pick the index
            auto criterio_checker = [&] {
                std::vector<ReturnType> result;
                bool is_complete;
                for (auto idx : basis_j_residual) {
                    is_complete = true;
                    if (EqualNotStrict(delta(idx, 0), 0)) {
                        is_complete &= x(idx, 0) <= matrix_ud(idx, 0) && x(idx, 0) >= matrix_ld(idx, 0);
                    } else if (delta(idx, 0) < 0) {
                        is_complete &= EqualNotStrict(x(idx, 0), matrix_ld(idx, 0));
                    } else  {
                        is_complete &= EqualNotStrict(x(idx, 0), matrix_ud(idx, 0));
                    }
                    if (!is_complete) {
                        result.emplace_back(idx, INDEX_TYPE::BOUND);
                    }
                }
    
                for (auto idx : basis_i) {
                    auto row_a = Matrix(matrix_a[idx]);
                    auto z = row_a * x;
                    is_complete = true;
                    if (EqualNotStrict(u(idx, 0), 0)) {
                        is_complete &= z(0, 0) <= matrix_ub(idx, 0) && z(0, 0) >= matrix_lb(idx, 0);
                    } else if (u(idx, 0) < 0) {
                        is_complete &= EqualNotStrict(z(0, 0), matrix_lb(idx, 0));
                    } else {
                        is_complete &= EqualNotStrict(z(0, 0), matrix_ub(idx, 0));
                    }

                    if (!is_complete) {
                        result.emplace_back(idx, INDEX_TYPE::VARIABLE);
                    }
                }
                return result;
            };

            auto bad_indexes = criterio_checker();
            if (bad_indexes.empty()) {
                return x;
            }

            auto [picked_index, type] = *bad_indexes.begin();

            auto lower_w = matrix_lb - matrix_a * x;
            auto upper_w = matrix_ub - matrix_a * x;


            // STEP 3 (v.3): compute beta

            auto compute_beta = [&] {
                double beta = 0;
                for (int idx = 0; idx < delta.GetRows(); idx++) {
                    if (delta(idx, 0) > 0) {
                        beta += delta(idx, 0) * (x(idx, 0) - matrix_ld(idx, 0));
                    } else {
                        beta += delta(idx, 0) * (x(idx, 0) - matrix_ud(idx, 0));
                    }
                }

                for (int idx = 0; idx < u.GetRows(); idx++) {
                    if (u(idx, 0) > 0) {
                        beta += u(idx, 0) * upper_w(idx, 0);
                    } else {
                        beta += u(idx, 0) * lower_w(idx, 0);
                    }
                }

                return beta;
            };

            auto beta = compute_beta();

            // STEP 4 (v. 3): compute l

            auto calculate_l = [&] {
                Matrix l(matrix_a.GetColumns(), 1, 0);
                for (int jdx = 0; jdx < l.GetRows(); jdx++) {
                    if (basis_j_residual.contains(jdx)) {
                        if (EqualNotStrict(delta(jdx, 0), 0)) {
                            continue;
                        }

                        if (delta(jdx, 0) < 0) {
                            l(jdx, 0) = matrix_ud(jdx, 0) - x(jdx, 0);
                        } else {
                            l(jdx, 0) = matrix_ld(jdx, 0) - x(jdx, 0);
                        }
                    } else {
                        auto matrix_a_basis_inv = matrix_a.GetSubMatrix(basis_i, basis_j).GetInversed();
                        
                        Matrix omega(basis_i.size(), 0);
                        int helper_idx = 0;
                        for (auto idx : basis_i) {
                            if (EqualNotStrict(u(jdx, 0), 0)) {
                                omega(helper_idx++, 0) = 0;
                                continue;
                            }
    
                            if (u(jdx, 0) < 0) {
                                omega(helper_idx++, 0) = lower_w(idx, 0);
                            } else {
                                omega(helper_idx++, 0) = upper_w(idx, 0);
                            }
                        }

                        auto l_sub = l.GetSubMatrix(basis_j_residual, {0});
                        auto matrix_roflo_submatrix = matrix_a.GetSubMatrix(basis_i, basis_j_residual);
                        auto lhs = matrix_a_basis_inv * omega;
                        auto rhs = matrix_a_basis_inv * matrix_roflo_submatrix * l_sub;

                        auto result = lhs - rhs;
                        helper_idx = 0;
                        for (auto idx : basis_i) {
                            l(idx, 0) = result(helper_idx++, 0);
                        }
                    }
                }
                return l;
            };

            auto l = calculate_l();

            // STEP 5 (v. 3): compute step (theta)

            double theta = 1;
            std::vector<Index> indexes;

            Index badass_index;

            for (auto jdx : basis_j) {
                indexes.emplace_back(jdx, INDEX_TYPE_PLAN::BASIS_J);
            }

            for (auto idx : basis_i_residual) {
                indexes.emplace_back(idx, INDEX_TYPE_PLAN::BASIS_I_INV);
            }

            // i can code it better, but who cares - 3:23 o'clock and i just want to this algo to FUCKING work
            for (auto [k, type] : indexes) {
                if (type == INDEX_TYPE_PLAN::BASIS_J) {
                    if (l(k, 0) > 0) {
                        double val = (matrix_ud(k, 0) - x(k, 0)) / l(k, 0);
                        if (theta > val) {
                            theta = val;
                            badass_index = {k , type};
                        }
                    } else {
                        double val = (matrix_ld(k, 0) - x(k, 0)) / l(k, 0);
                        if (theta > val) {
                            theta = val;
                            badass_index = {k , type};
                        }
                    }
                } else {
                    double interm_result = 0;
                    double interm_result_two = 0;
                    for (int jdx = 0; jdx < l.GetRows(); jdx++) {
                        interm_result += matrix_a(k, jdx) * l(jdx, 0);
                        interm_result_two += matrix_a(k, jdx) * x(jdx, 0);
                    }
                    if (EqualNotStrict(interm_result, 0)) {
                        continue;
                    }
                    if (interm_result > 0) {
                        double val = (matrix_ub(k, 0) - interm_result_two) / interm_result;
                        if (theta > val) {
                            theta = val;
                            badass_index = {k , type};
                        }
                    } else {
                        double val = (matrix_lb(k, 0) - interm_result_two) / interm_result;
                        if (theta > val) {
                            theta = val;
                            badass_index = {k , type};
                        }
                    }
                }
            }
            // TODO: before or after?
            auto kappa = x + l; 
            std::cout << "theta:\n";
            std::cout << theta << '\n';
            std::cout << "l:\n";
            std::cout << l;
            x = x + l * theta;
            
            std::cout << "x:\n";
            std::cout << x << '\n';

            std::cout << "target:\n";
            std::cout << matrix_c.GetTransposed() * x;

            auto beta_bar = (1. - theta) * beta;
            std::cout << "beta_bar: " << beta_bar << '\n';
            if (beta_bar <= kSolverTolerance) {
                return x;
            }

            // just printing some shit
            
            // STEP 9 (v 3.0)

            Matrix new_delta(basis_j_residual.size() + basis_i.size(), 1, 0);

            int dummy_idx = 0;
            for (auto jdx : basis_j_residual) {
                new_delta(dummy_idx++, 0) = delta(jdx, 0);
            }

            for (auto idx : basis_i) {
                new_delta(dummy_idx++, 0) = u(idx, 0);
            }

            delta = new_delta;

            Matrix new_kappa(basis_j_residual.size() + basis_i.size(), 1, 0);

            dummy_idx = 0;
            for (auto jdx : basis_j_residual) {
                new_kappa(dummy_idx++, 0) = x(jdx, 0) + l(jdx, 0);
            }

            for (auto idx : basis_i) {
                for (int jdx = 0; jdx < matrix_a.GetColumns(); jdx++) {
                    new_kappa(dummy_idx, 0) += matrix_a(idx, jdx) * (x(jdx, 0) + l(jdx, 0));
                }
                dummy_idx++;
            }

            kappa = new_kappa;

            auto compute_ksi_and_alpha_0 = [&] {
                Matrix ksi_j_res;
                Matrix ksi_i;
                Matrix ksi;
                
                double alpha_0;

                if (badass_index.type == INDEX_TYPE_PLAN::BASIS_J) {
                    auto matrix_a_basis_inv = matrix_a.GetSubMatrix(basis_i, basis_j).GetInversed();
                    auto matrix_roflo_submatrix = matrix_a.GetSubMatrix(basis_i, basis_j_residual);

                    if (matrix_a_basis_inv.GetRows() > 0 && matrix_a_basis_inv.GetColumns() > 0) {
                        ksi_j_res = matrix_a_basis_inv * matrix_roflo_submatrix;
                    }
                    
                    ksi_i = matrix_a_basis_inv * (-1);

                    ksi = Matrix::Stack(ksi_j_res, ksi_i);

                    int pos_idx = FindPosOfIndex(basis_j_residual, badass_index.idx);
                    double sign = EqualNotStrict(x(badass_index.idx, 0), matrix_ld(badass_index.idx, 0)) ? 1 : -1;
                    Matrix e(ksi.GetRows(), 1, 0);
                    e(pos_idx, 0) = sign;

                    auto transposed_ksi = e.GetTransposed() * ksi;
                    ksi = transposed_ksi.GetTransposed();

                    if (sign > 0) {
                        alpha_0 = kappa(badass_index.idx, 0) - matrix_ld(badass_index.idx, 0);
                    } else {
                        alpha_0 = -kappa(badass_index.idx, 0) + matrix_ud(badass_index.idx, 0);
                    }
                } else {
                    if (basis_i.empty() || basis_j.empty()) {
                        int dummy_idx = 0;
                        auto matrix_a_non_basis = matrix_a.GetSubMatrix(basis_i_residual, basis_j_residual);
                        ksi = matrix_a_non_basis;

                        double val = 0;
                        for (int idx = 0; idx < x.GetRows(); idx++) {
                            val += matrix_a(badass_index.idx, idx) * x(idx, 0);
                        }
                        double sign = EqualNotStrict(val, matrix_ub(badass_index.idx, 0)) ? 1 : -1;
                        
                        int pos_idx = FindPosOfIndex(basis_i_residual, badass_index.idx);

                        Matrix e(ksi.GetRows(), 1, 0);
                        e(pos_idx, 0) = sign;

                        auto transposed_ksi = e.GetTransposed() * ksi;
                        ksi = transposed_ksi.GetTransposed();

                        if (sign < 0) {
                            alpha_0 = kappa(badass_index.idx, 0) - matrix_lb(badass_index.idx, 0);
                        } else {
                            alpha_0 = -kappa(badass_index.idx, 0) + matrix_ub(badass_index.idx, 0);
                        }
                    } else {
                        auto matrix_a_basis_inv = matrix_a.GetSubMatrix(basis_i, basis_j).GetInversed();
                        auto matrix_a_non_basis = matrix_a.GetSubMatrix(basis_i_residual, basis_j_residual);
                        auto matrix_a_non_i_j = matrix_a.GetSubMatrix(basis_i_residual, basis_j);
                        auto matrix_roflo_submatrix = matrix_a.GetSubMatrix(basis_i, basis_j_residual);

                        ksi_j_res = matrix_a_non_basis;
                        auto interm_res = matrix_a_non_i_j * matrix_a_basis_inv * matrix_roflo_submatrix;
                        if (interm_res.GetRows() == basis_j_residual.size()) {
                            ksi_j_res -= interm_res;
                        }
                        ksi_i = matrix_a_non_i_j * matrix_a_basis_inv;
                        ksi = Matrix::Stack(ksi_j_res, ksi_i);

                        int pos_idx = FindPosOfIndex(basis_i_residual, badass_index.idx) + ksi_j_res.GetRows();
                        
                        double val = 0;
                        for (int idx = 0; idx < x.GetRows(); idx++) {
                            val += matrix_a(badass_index.idx, idx) * x(idx, 0);
                        }
                        double sign = EqualNotStrict(val, matrix_ub(badass_index.idx, 0)) ? 1 : -1;
                        
                        Matrix e(ksi.GetRows(), 1, 0);
                        e(pos_idx, 0) = sign;

                        auto transposed_ksi = e.GetTransposed() * ksi;
                        ksi = transposed_ksi.GetTransposed();

                        if (sign < 0) {
                            alpha_0 = kappa(badass_index.idx, 0) - matrix_lb(badass_index.idx, 0);
                        } else {
                            alpha_0 = -kappa(badass_index.idx, 0) + matrix_ub(badass_index.idx, 0);
                        }
                    }
                }

                if (ksi.GetRows() < basis_j_residual.size() + basis_i.size()) {
                    Matrix real_ksi(basis_j_residual.size() + basis_i.size(), 1, 0);
                    for (int idx = 0; idx < ksi.GetRows(); idx++) {
                        real_ksi(idx, 0) = ksi(idx, 0);
                    }
                    ksi = std::move(real_ksi);
                }
                return std::make_pair(ksi, alpha_0);
            };

            auto [ksi, alpha] = compute_ksi_and_alpha_0();

            // STEP 11 (v 3.0): some rofls
            // reminder: my delta has different sign!!!!

            // delta (J_n and I_op) = {delta (J_n), u(I_op)}
            // STEP 12 (v 3.0): compute sigma
            int sigma_idx = -1;
            INDEX_TYPE_PLAN_2 sigma_idx_type;
            if (alpha > 0) {
                std::cout << "what the fuck alpha > 0: " << alpha << '\n';
            }

            Matrix global_ud(basis_j_residual.size() + basis_i.size(), 1, 0);
            Matrix global_ld = global_ud;
            dummy_idx = 0;

            for (auto jdx : basis_j_residual) {
                global_ud(dummy_idx, 0) = matrix_ud(jdx, 0);
                global_ld(dummy_idx++, 0) = matrix_ld(jdx, 0);
            }

            for (auto idx : basis_i) {
                global_ud(dummy_idx, 0) = matrix_ub(idx, 0);
                global_ld(dummy_idx++, 0) = matrix_lb(idx, 0);
            }

            //

            std::cout << "ksi:\n";
            std::cout << ksi;
            std::cout << "delta:\n";
            std::cout << delta;
            std::cout << "global_ud:\n";
            std::cout << global_ud;
            std::cout << "global_ld:\n";
            std::cout << global_ld;
            std::cout << "kappa:\n";
            std::cout << kappa;
            std::cout << "alpha:\n";
            std::cout << alpha << '\n';

            do {
                for (int idx = 0; idx < ksi.GetRows(); idx++) {
                    double val;
                    if (ksi(idx, 0) < 0) {
                        val = ksi(idx, 0) * (kappa(idx, 0) - global_ud(idx, 0));
                        alpha += val;
                        if (val > 0) {
                            sigma_idx = idx;
                        }
                    } else {
                        val = ksi(idx, 0) * (kappa(idx, 0) - global_ld(idx, 0));
                        alpha += val;
                        if (val > 0) {
                            sigma_idx = idx;
                        }
                    }
                }

                delta = delta + ksi * alpha;

                // int dummy_idx = 0;

                // for (auto jdx : basis_j_residual) {
                //     if (delta(jdx, 0) * ksi(dummy_idx, 0) < 0) {
                //         double val = - delta(jdx, 0) / ksi(dummy_idx++, 0);
                //         if (val < sigma) {
                //             sigma = val;
                //             sigma_idx = jdx;
                //             sigma_idx_type = INDEX_TYPE_PLAN_2::BASIS_J_INV;
                //         }
                //     }
                // }

                // for (auto jdx : basis_i) {
                //     if (delta(jdx, 0) * ksi(dummy_idx, 0) < 0) {
                //         double val = - delta(jdx, 0) / ksi(dummy_idx++, 0);
                //         if (val < sigma) {
                //             sigma = val;
                //             sigma_idx = jdx;
                //             sigma_idx_type = INDEX_TYPE_PLAN_2::BASIS_I;
                //         }
                //     }
                // }

                // // STEP 13 (v 3.0)
                // dummy_idx = 0;
                // for (auto jdx : basis_j_residual) {
                //     delta(jdx, 0) += sigma * ksi(dummy_idx, 0);
                //     if (EqualNotStrict(delta(jdx, 0), 0)) {
                //         double val = ksi(dummy_idx, 0) > 0 ? matrix_ld(jdx, 0) : matrix_ud(jdx, 0);
                //         alpha += ksi(dummy_idx, 0) * (kappa(jdx, 0) - val);
                //     }
                //     dummy_idx++;
                // }

                // for (auto jdx : basis_i) {
                //     delta(jdx, 0) += sigma * ksi(dummy_idx, 0);
                //     if (EqualNotStrict(delta(jdx, 0), 0)) {
                //         double val = ksi(dummy_idx, 0) > 0 ? matrix_lb(jdx, 0) : matrix_ub(jdx, 0);
                //         alpha += ksi(dummy_idx, 0) * (kappa(jdx, 0) - val);
                //     }
                //     dummy_idx++;
                // }
            } while (alpha < 0);

            std::cout << "alpha:\n";
            std::cout << alpha << '\n';

            // STEP 16 (v. 3.0):

            // now we have sigma_idx and badass_index

            int bad_index = badass_index.idx;
            auto type_of_bad_index = badass_index.type;


            int bad_index_2;
            auto type_of_bad_index_2 = sigma_idx_type;

            if (sigma_idx >= basis_j_residual.size()) {
                bad_index_2 = sigma_idx - basis_j_residual.size();
                type_of_bad_index_2 = INDEX_TYPE_PLAN_2::BASIS_J_INV;
            } else {
                bad_index_2 = sigma_idx;
                type_of_bad_index_2 = INDEX_TYPE_PLAN_2::BASIS_I;
            }

            std::cout << "indexes: \n";
            std::cout << sigma_idx << '\n';
            std::cout << bad_index << " " << bad_index_2 << '\n';

            if (type_of_bad_index_2 == INDEX_TYPE_PLAN_2::BASIS_J_INV && type_of_bad_index == INDEX_TYPE_PLAN::BASIS_J) {
                basis_j.erase(bad_index);
                basis_j.insert(bad_index_2);

                basis_j_residual.erase(bad_index_2);
                basis_j_residual.insert(bad_index);
            } else if (type_of_bad_index_2 == INDEX_TYPE_PLAN_2::BASIS_J_INV && type_of_bad_index == INDEX_TYPE_PLAN::BASIS_I_INV) {
                basis_i.insert(bad_index);
                basis_j.insert(bad_index_2);

                basis_i_residual.erase(bad_index);
                basis_j_residual.erase(bad_index_2);
            } else if (type_of_bad_index_2 == INDEX_TYPE_PLAN_2::BASIS_I && type_of_bad_index == INDEX_TYPE_PLAN::BASIS_J) {
                basis_i.erase(bad_index_2);
                basis_j.erase(bad_index);

                basis_i_residual.insert(bad_index_2);
                basis_j_residual.insert(bad_index);
            } else {
                basis_i.erase(bad_index);
                basis_i.insert(bad_index_2);

                basis_i_residual.insert(bad_index);
                basis_i_residual.erase(bad_index_2);
            }

            // STEP 3.1 calculate (ksi) and (kappa)
            // auto calculate_ksi_and_kappa = [&](){
            //     // ksi (cpp), kappa (cev)
            //     Matrix ksi(matrix_a.GetRows(), 1, 0);
            //     Matrix kappa(matrix_a.GetColumns(), 1, 0);
            //     // page 86 (12.43), ...
            //     bool is_bad_case = false;

            //     auto matrix_a_basis = matrix_a.GetSubMatrix(basis_i, basis_j);
            //     auto matrix_a_basis_inv = matrix_a_basis.GetInversed();
            //     auto matrix_a_non_basis = matrix_a.GetSubMatrix(basis_i_residual, basis_j_residual);

            //     for (auto idx : basis_i) {
            //         if (EqualNotStrict(u(idx, 0), 0)) {
            //             is_bad_case = true;
            //         }
            //     }

            //     for (auto jdx : basis_j_residual) {
            //         if (EqualNotStrict(delta(jdx, 0), 0)) {
            //             is_bad_case = true;
            //         }
            //     }

            //     if (is_bad_case) {
            //         std::unordered_set<int> basis_i_zeros;
            //         std::unordered_set<int> basis_j_residual_zeros;
            //         for (int idx = 0; idx < u.GetRows(); idx++) {
            //             if (EqualNotStrict(u(idx, 0), 0)) {
            //                 basis_i_zeros.insert(idx);
            //             }
            //         }

            //         for (int jdx = 0; jdx < delta.GetRows(); jdx++) {
            //             if (EqualNotStrict(delta(jdx, 0), 0)) {
            //                 basis_j_residual_zeros.insert(jdx);
            //             }
            //         }

            //         if (!basis_i_zeros.empty()) {
            //             // pick one, and place it
            //             int picked = *basis_i_zeros.begin();
            //             basis_i_zeros.erase(picked);
            //             kappa(picked, 0) = matrix_ub(picked, 0);

            //             for (auto idx : basis_i_zeros) {
            //                 kappa(idx, 0) = matrix_lb(idx, 0);
            //             }
            //         }

            //         if (!basis_j_residual_zeros.empty()) {
            //             // pick one, and place it
            //             int picked = *basis_j_residual_zeros.begin();
            //             basis_j_residual_zeros.erase(picked);
            //             ksi(picked, 0) = matrix_ud(picked, 0);

            //             for (auto jdx : basis_j_residual_zeros) {
            //                 ksi(jdx, 0) = matrix_ld(jdx, 0);
            //             }
            //         }

            //         // if u(basis_i) > 0 => kappa() = ub
            //         // if u(basis_i) < 0 => kappa() = lb
            //         // if delta(basis_j_residual) > 0 => ud
            //         // if delta(basis_j_residual) < 0 => ld

            //         for (auto idx : basis_i) {
            //             if (EqualNotStrict(u(idx, 0), 0)) {
            //                 continue;
            //             }
            //             if (u(idx, 0) > 0) {
            //                 kappa(idx, 0) = matrix_ub(idx, 0);
            //             } else {
            //                 kappa(idx, 0) = matrix_lb(idx, 0);
            //             }
            //         }

            //         for (auto jdx : basis_j_residual) {
            //             if (EqualNotStrict(delta(jdx, 0), 0)) {
            //                 continue;
            //             }
            //             if (delta(jdx, 0) > 0) {
            //                 ksi(jdx, 0) = matrix_ud(jdx, 0);
            //             } else {
            //                 ksi(jdx, 0) = matrix_ld(jdx, 0);
            //             }
            //         }
            //     } else {
            //         for (auto idx : basis_i) {
            //             kappa(idx, 0) = (matrix_lb(idx, 0) + matrix_ub(idx, 0)) / 2;
            //         }

            //         for (auto idx : basis_i) {
            //             if (EqualNotStrict(u(idx, 0), 0)) {
            //                 continue;
            //             }
            //             if (u(idx, 0) > 0) {
            //                 kappa(idx, 0) = matrix_ub(idx, 0);
            //             } else {
            //                 kappa(idx, 0) = matrix_lb(idx, 0);
            //             }
            //         }

            //         for (auto jdx : basis_j_residual) {
            //             ksi(jdx, 0) = (matrix_ld(jdx, 0) + matrix_ud(jdx, 0)) / 2;
            //         }

            //         for (auto jdx : basis_j_residual) {
            //             if (EqualNotStrict(delta(jdx, 0), 0)) {
            //                 continue;
            //             }
            //             if (delta(jdx, 0) > 0) {
            //                 ksi(jdx, 0) = matrix_ud(jdx, 0);
            //             } else {
            //                 ksi(jdx, 0) = matrix_ld(jdx, 0);
            //             }
            //         }

            //         for (auto jdx : basis_j) {
            //             for (auto idx : basis_i) {
            //                 ksi(jdx, 0) += matrix_a_basis_inv(jdx, idx) * kappa(idx, 0);
            //                 for (auto jdx_inv : basis_j_residual) {
            //                     ksi(jdx, 0) -= matrix_a_basis_inv(jdx, idx) * matrix_a(idx, jdx_inv) * ksi(jdx_inv, 0);
            //                 }
            //             }
            //         }

            //         // kappa(jdx, 0) = A^(-1)(jdx, idx)ksi(idx, 0) - A^(-1)(jdx, idx) * A(idx, jdx_inv) * kappa(jdx_inv, 0)

            //         for (auto idx : basis_i_residual) {
            //             for (auto index = 0; index < matrix_a.GetColumns(); index++) {
            //                 kappa(idx, 0) += matrix_a(idx, index) * ksi(index, 0);
            //             }
            //         }
            //     }

            //     return std::make_tuple(std::move(ksi), std::move(kappa), is_bad_case);
            // };

            // // STEP 3.2: calculate (beta) - suboptimal cap
            // // intentionally swapped this guys! stay hard
            // auto [kappa, ksi, is_bad_case] = calculate_ksi_and_kappa();
            // std::cout << "ksi and kappa: \n";
            // std::cout << ksi << kappa;
            // auto calculate_beta = [&] {
            //     Matrix lhs = matrix_c.GetTransposed() * kappa;
            //     Matrix rhs = matrix_c.GetTransposed() * x;
            //     return lhs - rhs;
            // };

            // auto beta = calculate_beta()(0, 0);
            // // TODO: do smart return type!
            // std::cout << "beta: ";
            // std::cout << beta << '\n';
            // if (beta <= kSolverTolerance) {
            //     return x;
            // }

            // // STEP 4: build pseudoplan and calculate l = kappa - x
            // // kappa - pseudoplan, ksi - vector pseudospendings
            // auto l = kappa - x;

            // // STEP 5: compute maximum step (theta)
            // double theta = 1;
            // std::vector<Index> indexes;

            // for (auto jdx : basis_j) {
            //     indexes.emplace_back(jdx, INDEX_TYPE_PLAN::BASIS_J);
            // }

            // for (auto idx : basis_i_residual) {
            //     indexes.emplace_back(idx, INDEX_TYPE_PLAN::BASIS_I_INV);
            // }

            // for (auto [k, type] : indexes) {
            //     if (type == INDEX_TYPE_PLAN::BASIS_J) {
            //         if (l(k, 0) > 0) {
            //             theta = std::min(theta, (matrix_ud(k, 0) - x(k, 0)) / l(k, 0));
            //         } else {
            //             theta = std::min(theta, (matrix_ld(k, 0) - x(k, 0)) / l(k, 0));
            //         }
            //     } else {
            //         double interm_result = 0;
            //         double interm_result_two = 0;
            //         for (int jdx = 0; jdx < l.GetRows(); jdx++) {
            //             interm_result += matrix_a(k, jdx) * l(jdx, 0);
            //             interm_result_two += matrix_a(k, jdx) * x(jdx, 0);
            //         }
            //         if (EqualNotStrict(interm_result, 0)) {
            //             continue;
            //         }
            //         if (interm_result > 0) {
            //             theta = std::min(theta, (matrix_ub(k, 0) - interm_result_two) / interm_result);
            //         } else {
            //             theta = std::min(theta, (matrix_lb(k, 0) - interm_result_two) / interm_result);
            //         }
            //     }
            // }
            // std::cout << "l:\n";
            // std::cout << l;
            // std::cout << "theta: " << theta << '\n';
            // // STEP 6: compute new plan x_new = x + theta * l
            // x = x + l * theta;
            // std::cout << "x:\n";
            // std::cout << x << '\n';

            // // STEP 7: ocenka suboptimalnosty
            // auto beta_bar = (1. - theta) * beta;
            // std::cout << "beta_bar: " << beta_bar << '\n';
            // if (beta_bar <= kSolverTolerance) {
            //     return x;
            // }
            // // return Matrix{};
            // // STEP 8.1: get bad indexes 
            
            // auto check_dummy_criterion = [&]() {
            //     std::vector<Index> vector;
            //     for (int jdx : basis_j) {
            //         if (kappa(jdx, 0) < matrix_ld(jdx, 0) || kappa(jdx, 0) > matrix_ud(jdx, 0)) {
            //             vector.emplace_back(jdx, INDEX_TYPE_PLAN::BASIS_J);
            //         }
            //     }

            //     for (int idx : basis_i_residual) {
            //         if (ksi(idx, 0) < matrix_lb(idx, 0) || ksi(idx, 0) > matrix_ub(idx, 0)) {
            //             vector.emplace_back(idx, INDEX_TYPE_PLAN::BASIS_I_INV);
            //         }
            //     }

            //     return vector;
            // };
            // auto bad_indexes_dummy = check_dummy_criterion();
            // if (bad_indexes_dummy.empty()) {
            //     // if (!basis_j.empty()) {
            //     //     bad_indexes_dummy.push_back({*basis_j.begin(), INDEX_TYPE_PLAN::BASIS_J});
            //     // } else if (!basis_i_residual.empty()) {
            //     //     bad_indexes_dummy.push_back({*basis_i_residual.begin(), INDEX_TYPE_PLAN::BASIS_I_INV});
            //     // } else {
            //     //     std::cout << "bad_indexes_dummy is empty!\n";
            //     //     return kappa;
            //     // }
            //     std::cout << "bad_indexes_dummy is empty!\n";
            //     return kappa;
            // }
            // auto [bad_index, type_of_bad_index] = *bad_indexes_dummy.begin();

            // // WRITING ANOTHER VERSION!!!!!\n
            // // we have u, delta, ksi (cpp), kappa (cev)
            // // bad_index (number), type_of_bad_index (direct_ind)
            // // bad_case (degenerate_case)
            // // ref : page 113

            // auto shift_basis = [&] {
            //     auto matrix_a_basis = matrix_a.GetSubMatrix(basis_i, basis_j);
            //     auto matrix_a_basis_inv = matrix_a_basis.GetInversed();
            //     auto matrix_a_non_basis = matrix_a.GetSubMatrix(basis_i_residual, basis_j_residual);

            //     Matrix delta_d(matrix_a.GetColumns(), 1, 0);
            //     Matrix delta_y(matrix_a.GetRows(), 1, 0);

            //     if (!basis_i.empty()) {
            //         double g;
            //         if (type_of_bad_index == INDEX_TYPE_PLAN::BASIS_I_INV) {
            //             g = ksi(bad_index, 0) - x(bad_index, 0);
            //             delta_d(bad_index, 0) = g > 0 ? 1 : -1;
            //             auto matrix_a_basis_inv_transposed = matrix_a_basis_inv.GetTransposed();

            //             auto delta_d_sub = delta_d.GetSubMatrix(basis_j, {0});
            //             auto res = matrix_a_basis_inv_transposed * delta_d_sub;

            //             int helper_idx = 0;
            //             for (auto idx : basis_i) {
            //                 delta_y(idx, 0) -= res(helper_idx++, 0);
            //                 // for (auto jdx : basis_j) {
            //                 //     delta_y(idx, 0) -= matrix_a_basis_inv_transposed(jdx, idx) * delta_d(jdx, 0);
            //                 // }
            //             }

            //             auto matrix_a_non_basis_transposed = matrix_a_non_basis.GetTransposed();
            //             auto delta_y_sub = delta_y.GetSubMatrix(basis_i, {0});
            //             res = matrix_a_non_basis_transposed * delta_y_sub;
            //             helper_idx = 0;
            //             for (auto jdx : basis_j_residual) {
            //                 delta_d(jdx, 0) -= res(helper_idx++, 0);
            //                 // for (auto idx : basis_i) {
            //                 //     delta_d(jdx, 0) -= matrix_a_non_basis_transposed(jdx, idx) * delta_y(idx, 0);
            //                 // }
            //             }
            //         } else {
            //             auto z = matrix_a * x;
            //             g = kappa(bad_index, 0) - z(bad_index, 0);
            //             double sign = g > 0 ? 1 : -1;
            //             auto matrix_a_basis_inv_transposed = matrix_a_basis_inv.GetTransposed();
            //             auto matrix_a_transposed = matrix_a.GetTransposed();
            //             delta_y(bad_index, 0) = g > 0 ? 1 : -1;

            //             auto matrix_a_transposed_submatrix = matrix_a_transposed.GetSubMatrix(basis_j, {bad_index});
            //             auto result = matrix_a_basis_inv.GetTransposed() * matrix_a_transposed_submatrix;
            //             int helper_idx = 0;
            //             for (auto idx : basis_i) {
            //                 delta_y(idx, 0) += -sign * result(helper_idx++, 0); 
            //             }

            //             for (auto jdx : basis_j_residual) {
            //                 for (auto idx = 0; idx < delta_y.GetRows(); idx++) {
            //                     delta_d(jdx, 0) -= delta_y(idx, 0) * matrix_a(idx, jdx);
            //                 }
            //             }
            //         }

            //         double alpha = -std::abs(g);
            //         Matrix s_i(delta_y.GetRows(), 1, 0);
            //         Matrix s_j(delta_d.GetRows(), 1, 0);

            //         std::unordered_set<int> basis_i_zeros;
            //         std::unordered_set<int> basis_j_residual_zeros;
            //         if (is_bad_case) {
            //             for (auto idx : basis_i) {
            //                 if (EqualNotStrict(u(idx, 0), 0)) {
            //                     basis_i_zeros.insert(idx);
            //                     alpha += (matrix_ub(idx, 0) - matrix_lb(idx, 0)) * delta_y(idx, 0);
            //                 }
            //             }
            //             for (auto jdx : basis_j_residual) {
            //                 if (EqualNotStrict(jdx, 0)) {
            //                     basis_j_residual_zeros.insert(jdx);
            //                     alpha += (matrix_ud(jdx, 0) - matrix_ld(jdx, 0)) * delta_d(jdx, 0);
            //                 }
            //             }
            //         }
            //         int roflo_idx;
            //         INDEX_TYPE_PLAN_2 roflo_idx_type;
            //         if (alpha >= 0 && is_bad_case) {
            //             if (!basis_i_zeros.empty()) {
            //                 roflo_idx_type = INDEX_TYPE_PLAN_2::BASIS_I;
            //                 roflo_idx = *basis_i_zeros.begin();
            //             } else {
            //                 roflo_idx_type = INDEX_TYPE_PLAN_2::BASIS_J_INV;
            //                 roflo_idx = *basis_j_residual_zeros.begin();
            //             }
            //         } else {
            //             for (int idx = 0; idx < s_i.GetRows(); idx++) {
            //                 s_i(idx, 0) = 1e9; // inf
            //                 if (basis_i.contains(idx)) {
            //                     if (u(idx, 0) * delta_y(idx, 0) < 0) {
            //                         s_i(idx, 0) = -u(idx, 0) / delta_y(idx, 0);
            //                     }
            //                 }
            //             }

            //             for (int jdx = 0; jdx < s_j.GetRows(); jdx++) {
            //                 s_j(jdx, 0) = 1e9; //inf
            //                 if (basis_j.contains(jdx)) {
            //                     if (delta(jdx, 0) * delta_d(jdx, 0) < 0) {
            //                         s_j(jdx,0) = - delta(jdx, 0) / delta_d(jdx, 0);
            //                     }
            //                 }
            //             }

            //             double min_idx_val = 1e9;
            //             int min_idx = 0;

            //             double min_jdx_val = 1e9;
            //             int min_jdx = 0;

            //             for (int idx = 0; idx < s_i.GetRows(); idx++) {
            //                 if (min_idx_val > s_i(idx, 0)) {
            //                     min_idx = idx;
            //                     min_idx_val = s_i(idx, 0);
            //                 }
            //             }

            //             for (int jdx = 0; jdx < s_j.GetRows(); jdx++) {
            //                 if (min_jdx_val > s_j(jdx, 0)) {
            //                     min_jdx = jdx;
            //                     min_jdx_val = s_j(jdx, 0);
            //                 }
            //             }

            //             if (min_idx_val > 1e8 && min_jdx_val > 1e8) {
            //                 std::cout << "all the values are fucked bro!\n";
            //             }

            //             if (min_idx_val < min_jdx_val) {
            //                 roflo_idx = min_idx;
            //                 roflo_idx_type = INDEX_TYPE_PLAN_2::BASIS_I;
            //             } else {
            //                 roflo_idx = min_jdx;
            //                 roflo_idx_type = INDEX_TYPE_PLAN_2::BASIS_J_INV;
            //             }
            //         }
            //         // true => basis_i_inv
            //         // false => basis_j
            //         // roflo_idx_type == 1 => INDEX_TYPE_PLAN_2::BASIS_I
            //         // roflo_idx_type == 2 => INDEX_TYPE_PLAN_2::BASIS_J_INV
            //         if (type_of_bad_index == INDEX_TYPE_PLAN::BASIS_J && roflo_idx_type == INDEX_TYPE_PLAN_2::BASIS_I) {
            //             basis_i.insert(bad_index);
            //             basis_i.erase(roflo_idx);

            //             basis_i_residual.insert(roflo_idx);
            //             basis_i_residual.erase(bad_index);
            //         } else if (type_of_bad_index == INDEX_TYPE_PLAN::BASIS_J && roflo_idx_type == INDEX_TYPE_PLAN_2::BASIS_J_INV) {
            //             basis_i.insert(bad_index);
            //             basis_j.insert(roflo_idx);

            //             basis_i_residual.erase(bad_index);
            //             basis_j_residual.erase(roflo_idx);
            //         } else if (type_of_bad_index == INDEX_TYPE_PLAN::BASIS_I_INV && roflo_idx_type == INDEX_TYPE_PLAN_2::BASIS_I) {
            //             basis_i.erase(roflo_idx);
            //             basis_j.erase(bad_index);

            //             basis_i_residual.insert(roflo_idx);
            //             basis_j_residual.insert(bad_index);
            //         } else {
            //             basis_j.insert(roflo_idx);
            //             basis_j.erase(bad_index);

            //             basis_j_residual.erase(roflo_idx);
            //             basis_j_residual.insert(bad_index);
            //         }
            //     } else {
            //         auto z = matrix_a * x;
            //         auto g = kappa(bad_index, 0) - z(bad_index, 0);
            //         delta_y(bad_index, 0) = g > 0 ? 1 : -1;
                    
            //         auto interm_res = matrix_a.GetTransposed() * delta_y;
            //         interm_res *= -1;
            //         for (int idx = 0; idx < interm_res.GetRows(); idx++) {
            //             delta_d(idx, 0) = interm_res(idx, 0);
            //         }
            //         double alpha = -std::abs(g);

            //         std::unordered_set<int> basis_i_zeros;
            //         std::unordered_set<int> basis_j_residual_zeros;

            //         if (is_bad_case) {
            //             for (auto jdx : basis_j_residual) {
            //                 if (EqualNotStrict(jdx, 0)) {
            //                     basis_j_residual_zeros.insert(jdx);
            //                     alpha += (matrix_ud(jdx, 0) - matrix_ld(jdx, 0)) * delta_d(jdx, 0);
            //                 }
            //             }
            //         }
            //         Matrix s_i(delta_y.GetRows(), 1, 0);
            //         Matrix s_j(delta_d.GetRows(), 1, 0);

            //         int roflo_idx;
            //         if (alpha >= 0 && is_bad_case) {
            //             roflo_idx = *basis_j_residual_zeros.begin();
            //         } else {
            //             double min_s_j = 1e9;
            //             for (int jdx = 0; jdx < delta_d.GetRows(); jdx++) {
            //                 s_j(jdx, 0) = 1e9; //inf
            //                 if (delta(jdx, 0) * delta_d(jdx, 0) < 0) {
            //                     s_j(jdx, 0) = -delta(jdx, 0) / delta_d(jdx,0);
            //                     if (s_j(jdx, 0) < min_s_j) {
            //                         min_s_j = s_j(jdx, 0);
            //                         roflo_idx = jdx;
            //                     }
            //                 }
            //             }

            //             if (min_s_j > 1e8) {
            //                 std::cout << "too big step\n";
            //             }
            //         }

            //         basis_i.insert(bad_index);
            //         basis_j.insert(roflo_idx);

            //         basis_i_residual.erase(bad_index);
            //         basis_j_residual.erase(roflo_idx);
            //     }
            // };

            // shift_basis();


            // STEP 8.2: build gradient for deltas

            
            // std::unordered_set<int> basis_i; // from 0 to matrix_a.GetRows() - 1
            // std::unordered_set<int> basis_j; // from 0 to matrix_a.GetColumns() - 1
            

            // Matrix delta_d(matrix_a.GetColumns(), 1, 0);
            // Matrix delta_y(matrix_a.GetRows(), 1, 0); 
            
            // if (type_of_bad_index == INDEX_TYPE_PLAN::BASIS_J) {
            //     delta_d(bad_index, 0) = kappa(bad_index, 0) - matrix_ld(bad_index, 0) > 0 ? 1 : -1;
            //     // delta_d(basis_j, 0) = 0
            //     // delta_y(basis_i_inv, 0) = 0

            //     auto matrix_a_inversed = matrix_a.GetSubMatrix(basis_i, basis_j).GetInversed();

            //     for (auto idx : basis_i) {
            //         for (auto jdx : basis_j) {
            //             delta_y(idx, 0) -= delta_d(jdx, 0) * matrix_a_inversed(jdx, idx);
            //         }
            //     }

            //     auto matrix_a_transposed = matrix_a.GetSubMatrix(basis_i, basis_j).GetTransposed();

            //     for (auto jdx : basis_j_residual) {
            //         for (auto idx : basis_i) {
            //             delta_d(jdx, 0) -= matrix_a_transposed(idx, jdx) * delta_y(idx, 0);
            //         }
            //     }
            // } else {
            //     delta_y(bad_index, 0) = ksi(bad_index, 0) - matrix_lb(bad_index, 0) > 0 ? 1 : -1;
            //     // delta_y(basis_i_inv, 0) = 0
            //     // delta_d(basis_j, 0) = 0

            //     auto matrix_a_inversed = matrix_a.GetSubMatrix(basis_i, basis_j).GetInversed();
            //     auto matrix_a_transposed = matrix_a.GetSubMatrix(basis_i, basis_j).GetTransposed();

            //     for (auto idx : basis_i) {
            //         for (auto jdx : basis_j) {
            //             double res = 0;
            //             for (auto idx_inv : basis_i_residual) {
            //                 res += delta_y(idx_inv, 0) * matrix_a(idx_inv, jdx);
            //             }
            //             delta_y(idx, 0) += res * matrix_a_inversed(jdx, idx);
            //         }
            //     }

            //     for (auto jdx_inv : basis_j_residual) {
            //         for (int idx = 0; idx < matrix_a.GetRows(); idx++) {
            //             delta_d(jdx_inv, 0) -= matrix_a_transposed(idx, jdx_inv) * delta_y(idx, 0);
            //         }
            //     }
            // }
            // int bad_index_2 = -1;
            // INDEX_TYPE_PLAN_2 type_of_bad_index_2;
            // // STEP 9: compute short! step
            // double sigma_step = 1e9; // ?
            // for (int k : basis_j_residual) {
            //     if ((delta_d(k, 0) < 0 && kappa(k, 0) > matrix_ld(k, 0)) || (delta_d(k, 0) > 0 && kappa(k, 0) < matrix_ud(k, 0))) {
            //         double value = -delta(k, 0) / delta_d(k, 0);
            //         if (sigma_step > value) {
            //             sigma_step = value;
            //             bad_index_2 = k;
            //             type_of_bad_index_2 = INDEX_TYPE_PLAN_2::BASIS_J_INV;
            //         }
            //     }
            // }

            // // maybe here not basis_j, but basis_i ?
            // // upd: changed basis_j to basis_i
        
            // for (int k : basis_i) { 
            //     // here some delta_u, not u!!!!
            //     // upd: changed u to delta_y
            //     if ((delta_y(k, 0) < 0 && ksi(k, 0) > matrix_lb(k, 0)) || (delta_y(k, 0) > 0 && ksi(k, 0) < matrix_ub(k, 0))) {
            //         double value = -u(k, 0) / delta_y(k, 0);
            //         if (sigma_step > value) {
            //             sigma_step = value;
            //             bad_index_2 = k;
            //             type_of_bad_index_2 = INDEX_TYPE_PLAN_2::BASIS_I;
            //         }
            //     }
            // }
            
            // if (sigma_step > 1e8) {
            //     std::cout << "sigma is too big!\n";
            // }

            // // STEP 10: changing opora
            // // working with k_0 (bad_index_2), and k_* (bad_index)

            // if (type_of_bad_index_2 == INDEX_TYPE_PLAN_2::BASIS_J_INV && type_of_bad_index == INDEX_TYPE_PLAN::BASIS_J) {
            //     basis_j.erase(bad_index);
            //     basis_j.insert(bad_index_2);

            //     basis_j_residual.erase(bad_index_2);
            //     basis_j_residual.insert(bad_index);
            // } else if (type_of_bad_index_2 == INDEX_TYPE_PLAN_2::BASIS_J_INV && type_of_bad_index == INDEX_TYPE_PLAN::BASIS_I_INV) {
            //     basis_i.insert(bad_index);
            //     basis_j.insert(bad_index_2);

            //     basis_i_residual.erase(bad_index);
            //     basis_j_residual.erase(bad_index_2);
            // } else if (type_of_bad_index_2 == INDEX_TYPE_PLAN_2::BASIS_I && type_of_bad_index == INDEX_TYPE_PLAN::BASIS_J) {
            //     basis_i.erase(bad_index_2);
            //     basis_j.erase(bad_index);

            //     basis_i_residual.insert(bad_index_2);
            //     basis_j_residual.insert(bad_index);
            // } else {
            //     basis_i.erase(bad_index);
            //     basis_i.insert(bad_index_2);

            //     basis_i_residual.insert(bad_index);
            //     basis_i_residual.erase(bad_index_2);
            // }
            // // go to STEP 1
        }
    }
};