#pragma once
#include "alglib/optimization.h"
#include "transformer.h"

#include <unordered_set>
#include <ranges>

constexpr double kSolverTolerance = 1e0;

class Solver {
    enum class INDEX_TYPE { BOUND, VARIABLE };
    enum class INDEX_TYPE_PLAN { BASIS_J, BASIS_I_INV };
    enum class INDEX_TYPE_PLAN_2 { BASIS_J_INV, BASIS_I };
public:
    struct ReturnType {
        int idx;
        INDEX_TYPE type;
    };

    struct Index {
        int idx;
        INDEX_TYPE_PLAN type;
    };

    struct Index2 {
        int idx;
        INDEX_TYPE_PLAN_2 type;
    };

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
        const Matrix matrix_a_transposed = matrix_a.GetTransposed();
        // STEP 0: CHOOSING OPORNIY PLAN
        auto x = (matrix_ld + matrix_ud) / 2;
        // TODO: rofls
        x = Matrix{{0, 2}};
        x = x.GetTransposed();
        std::unordered_set<int> basis_i; // from 0 to matrix_a.GetRows() - 1
        std::unordered_set<int> basis_j; // from 0 to matrix_a.GetColumns() - 1

        std::unordered_set<int> basis_i_residual;
        for (int idx = 0; idx < matrix_a.GetRows(); idx++) {
            basis_i_residual.insert(idx);
        }

        std::unordered_set<int> basis_j_residual;
        for (int idx = 0; idx < matrix_a.GetColumns(); idx++) {
            basis_j_residual.insert(idx);
        }

        // setting criterio checker
        auto criterio_checker = [&](const Matrix& u, const Matrix& delta) {
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
                auto z = row_a * x.GetTransposed();
                is_complete = true;
                if (EqualNotStrict(u(idx, 0), 0)) {
                    is_complete &= z(0, 0) <= matrix_ub(idx, 0) && z(0, 0) >= matrix_lb(idx, 0);
                } else if (u(idx, 0) < 0) {
                    is_complete &= EqualNotStrict(z(0, 0), matrix_lb(idx, 0));
                } else {
                    is_complete &= EqualNotStrict(z(0, 0), matrix_ub(idx, 0));
                }
                result.emplace_back(idx, INDEX_TYPE::VARIABLE);
            }
            return result;
        };

        while (true) {
            // STEP 1: compute (u) and (\delta) vector
            auto calculate_potential_delta = [&]{
                Matrix u(matrix_a.GetRows(), 1, 0);
                Matrix delta(matrix_a.GetColumns(), 1, 0);
                // u^T(idx) = c^T(jdx)A^(-1)(jdx, idx) for all idx
                auto matrix_a_inversed = matrix_a.GetSubMatrix(basis_i, basis_j).GetInversed();
                for (auto idx : basis_i) {
                    for (auto jdx : basis_j) {
                        // TODO: intentionally changed (jdx, idx) to (idx, jdx), makes no sense of the dimentionality!
                        // changed back because of the rows == cols, and i think i need to compute A^(-1) for CERTAIN indexes!
                        u(idx, 0) += matrix_c(jdx, 0) * matrix_a_inversed(jdx, idx);
                    }
                }

                // delta(jdx_inv) = c(jdx_inv) - A^T(idx, jdx_inv)u(idx) for all jdx
                for (auto jdx : basis_j_residual) {
                    delta(jdx, 0) += matrix_c(jdx, 0);
                    for (auto idx : basis_i) {
                        delta(jdx, 0) -= matrix_a_transposed(idx, jdx) * u(idx, 0); 
                    }
                }

                return std::make_pair(std::move(u), std::move(delta));
            };

            auto [u, delta] = calculate_potential_delta();

            // STEP 2: checking criterio, if return array is empty, then stop the execution
            // otherwise, pick the index
            auto bad_indexes = criterio_checker(u, delta);
            if (bad_indexes.empty()) {
                return x;
            }

            auto [picked_index, type] = *bad_indexes.begin();
            for (auto u : bad_indexes) {
                std::cout << u.idx << '\n';
            }
            // STEP 3.1 calculate (ksi) and (kappa)
            auto calculate_ksi_and_kappa = [&](){
                Matrix ksi(matrix_a.GetRows(), 1, 0);
                Matrix kappa(matrix_a.GetColumns(), 1, 0);
                // page 86 (12.43), ...

                for (int idx = 0; idx < ksi.GetRows(); idx++) {
                    // all of matrix_i are these guys!
                    if (EqualNotStrict(u(idx, 0), 0)) {
                        continue;
                    }

                    if (u(idx, 0) < 0) {
                        ksi(idx, 0) = matrix_lb(idx, 0);
                    } else {
                        ksi(idx, 0) = matrix_ub(idx, 0);
                    }
                }
                
                for (int jdx = 0; jdx < kappa.GetRows(); jdx++) {
                    // all of matrix_j_residual are these guys!
                    if (EqualNotStrict(delta(jdx, 0), 0)) {
                        continue;
                    }

                    if (delta(jdx, 0) < 0) {
                        kappa(jdx, 0) = matrix_ld(jdx, 0);
                    } else {
                        kappa(jdx, 0) = matrix_ud(jdx, 0);
                    }
                }
                // TODO: basis_i, basis_j can be empty
                auto matrix_a_inversed = matrix_a.GetSubMatrix(basis_i, basis_j).GetInversed();
                // FUCK
                // kappa(jdx, 0) = A^(-1)(jdx, idx)ksi(idx, 0) - A^(-1)(jdx, idx) * A(idx, jdx_inv) * kappa(jdx_inv, 0)
                for (auto jdx : basis_j) {
                    for (auto idx : basis_i) {
                        kappa(jdx, 0) += matrix_a_inversed(jdx, idx) * ksi(idx, 0);
                        for (auto jdx_inv : basis_j_residual) {
                            kappa(jdx, 0) -= matrix_a_inversed(jdx, idx) * matrix_a(idx, jdx_inv) * kappa(jdx_inv, 0);
                        }
                    }
                }

                for (auto idx : basis_i_residual) {
                    for (auto index = 0; index < matrix_a.GetColumns(); index++) {
                        ksi(idx, 0) += matrix_a(idx, index) * kappa(index, 0);
                    }
                }
                return std::make_pair(std::move(ksi), std::move(kappa));
            };

            // STEP 3.2: calculate (beta) - suboptimal cap
            auto [ksi, kappa] = calculate_ksi_and_kappa();
            std::cout << "ksi and kappa: \n";
            std::cout << ksi << kappa;
            // auto calculate_beta = [&](){
            //     double beta = 0;
            //     for (auto jdx_inv : basis_j_residual) {
            //         if (delta(jdx_inv, 0) < 0) {
            //             beta += delta(jdx_inv, 0) * (matrix_ld(jdx_inv, 0) - kappa(jdx_inv, 0));
            //         } else {
            //             beta += delta(jdx_inv, 0) * (matrix_ud(jdx_inv, 0) - kappa(jdx_inv, 0));
            //         }
            //     }

            //     for (auto idx : basis_i) {
            //         if (u(idx, 0) < 0) {
            //             beta += u(idx, 0) * (matrix_lb(idx, 0) - ksi(idx, 0));
            //         } else {
            //             beta += u(idx, 0) * (matrix_ub(idx, 0) - ksi(idx, 0));
            //         }
            //     }
            //     return beta;
            // };
            auto calculate_beta = [&] {
                Matrix lhs = matrix_c * kappa;
                Matrix rhs = matrix_c * x;
                return lhs - rhs;
            };

            auto beta = calculate_beta()(0, 0);
            // TODO: do smart return type!
            std::cout << "beta: ";
            std::cout << beta << '\n';
            if (beta <= kSolverTolerance) {
                return x;
            }

            // STEP 4: build pseudoplan and calculate l = kappa - x
            // kappa - pseudoplan, ksi - vector pseudospendings
            auto l = kappa - x;

            // STEP 5: compute maximum step (theta)
            double theta = 1;
            std::vector<Index> indexes;

            for (auto jdx : basis_j) {
                indexes.emplace_back(jdx, INDEX_TYPE_PLAN::BASIS_J);
            }

            for (auto idx : basis_i_residual) {
                indexes.emplace_back(idx, INDEX_TYPE_PLAN::BASIS_I_INV);
            }

            for (auto [k, type] : indexes) {
                if (type == INDEX_TYPE_PLAN::BASIS_J) {
                    if (l(k, 0) > 0) {
                        theta = std::min(theta, (matrix_ud(k, 0) - x(k, 0)) / l(k, 0));
                    } else {
                        theta = std::min(theta, (matrix_ld(k, 0) - x(k, 0)) / l(k, 0));
                    }
                } else {
                    double interm_result = 0;
                    double interm_result_two = 0;
                    for (int jdx = 0; jdx < l.GetRows(); jdx++) {
                        interm_result += matrix_a(k, jdx) * l(jdx, 0);
                        interm_result_two += matrix_a(k, jdx) * x(jdx, 0);
                    }

                    if (interm_result > 0) {
                        theta = std::min(theta, (matrix_ub(k, 0) - interm_result_two) / interm_result);
                    } else {
                        theta = std::min(theta, (matrix_lb(k, 0) - interm_result_two) / interm_result);
                    }
                }
            }

            // STEP 6: compute new plan x_new = x + theta * l
            x = x + l * theta;

            // STEP 7: ocenka suboptimalnosty
            auto beta_bar = (1. - theta) * beta;
            if (beta_bar <= kSolverTolerance) {
                return x;
            }

            // WRITING ANOTHER VERSION!!!!!\n

            // STEP 8.1: get bad indexes 
            
            auto check_dummy_criterion = [&]() {
                std::vector<Index> vector;
                for (int jdx : basis_j) {
                    if (kappa(jdx, 0) < matrix_ld(jdx, 0) || kappa(jdx, 0) > matrix_ud(jdx, 0)) {
                        vector.emplace_back(jdx, INDEX_TYPE_PLAN::BASIS_J);
                    }
                }

                for (int idx : basis_i_residual) {
                    if (ksi(idx, 0) < matrix_lb(idx, 0) || ksi(idx, 0) > matrix_ub(idx, 0)) {
                        vector.emplace_back(idx, INDEX_TYPE_PLAN::BASIS_J);
                    }
                }

                return vector;
            };
            auto bad_indexes_dummy = check_dummy_criterion();
            auto [bad_index, type_of_bad_index] = *bad_indexes_dummy.begin();

            // STEP 8.2: build gradient for deltas

            /*
            std::unordered_set<int> basis_i; // from 0 to matrix_a.GetRows() - 1
            std::unordered_set<int> basis_j; // from 0 to matrix_a.GetColumns() - 1
            */

            Matrix delta_d(matrix_a.GetColumns(), 1, 0);
            Matrix delta_y(matrix_a.GetRows(), 1, 0); 
            
            if (type_of_bad_index == INDEX_TYPE_PLAN::BASIS_J) {
                delta_d(bad_index, 0) = kappa(bad_index, 0) - matrix_ld(bad_index, 0) > 0 ? 1 : -1;
                // delta_d(basis_j, 0) = 0
                // delta_y(basis_i_inv, 0) = 0

                auto matrix_a_inversed = matrix_a.GetSubMatrix(basis_i, basis_j).GetInversed();

                for (auto idx : basis_i) {
                    for (auto jdx : basis_j) {
                        delta_y(idx, 0) -= delta_d(jdx, 0) * matrix_a_inversed(jdx, idx);
                    }
                }

                auto matrix_a_transposed = matrix_a.GetSubMatrix(basis_i, basis_j).GetTransposed();

                for (auto jdx : basis_j_residual) {
                    for (auto idx : basis_i) {
                        delta_d(jdx, 0) -= matrix_a_transposed(idx, jdx) * delta_y(idx, 0);
                    }
                }
            } else {
                delta_y(bad_index, 0) = ksi(bad_index, 0) - matrix_lb(bad_index, 0) > 0 ? 1 : -1;
                // delta_y(basis_i_inv, 0) = 0
                // delta_d(basis_j, 0) = 0

                auto matrix_a_inversed = matrix_a.GetSubMatrix(basis_i, basis_j).GetInversed();
                auto matrix_a_transposed = matrix_a.GetSubMatrix(basis_i, basis_j).GetTransposed();

                for (auto idx : basis_i) {
                    for (auto jdx : basis_j) {
                        double res = 0;
                        for (auto idx_inv : basis_i_residual) {
                            res += delta_y(idx_inv, 0) * matrix_a(idx_inv, jdx);
                        }
                        delta_y(idx, 0) += res * matrix_a_inversed(jdx, idx);
                    }
                }

                for (auto jdx_inv : basis_j_residual) {
                    for (int idx = 0; idx < matrix_a.GetRows(); idx++) {
                        delta_d(jdx_inv, 0) -= matrix_a_transposed(idx, jdx_inv) * delta_y(idx, 0);
                    }
                }
            }
            int bad_index_2 = -1;
            INDEX_TYPE_PLAN_2 type_of_bad_index_2;
            // STEP 9: compute short! step
            double sigma_step = 1e9; // ?
            for (int k : basis_j_residual) {
                if ((delta_d(k, 0) < 0 && kappa(k, 0) > matrix_ld(k, 0)) || (delta_d(k, 0) > 0 && kappa(k, 0) < matrix_ud(k, 0))) {
                    double value = -delta(k, 0) / delta_d(k, 0);
                    if (sigma_step > value) {
                        sigma_step = value;
                        bad_index_2 = k;
                        type_of_bad_index_2 = INDEX_TYPE_PLAN_2::BASIS_J_INV;
                    }
                }
            }

            // maybe here not basis_j, but basis_i ?
            // upd: changed basis_j to basis_i
        
            for (int k : basis_i) { 
                // here some delta_u, not u!!!!
                // upd: changed u to delta_y
                if ((delta_y(k, 0) < 0 && ksi(k, 0) > matrix_lb(k, 0)) || (delta_y(k, 0) > 0 && ksi(k, 0) < matrix_ub(k, 0))) {
                    double value = -u(k, 0) / delta_y(k, 0);
                    if (sigma_step > value) {
                        sigma_step = value;
                        bad_index_2 = k;
                        type_of_bad_index_2 = INDEX_TYPE_PLAN_2::BASIS_I;
                    }
                }
            }
            
            if (sigma_step > 1e8) {
                std::cout << "sigma is too big!\n";
            }

            // STEP 10: changing opora
            // working with k_0 (bad_index_2), and k_* (bad_index)

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
            // go to STEP 1
        }
    }
};