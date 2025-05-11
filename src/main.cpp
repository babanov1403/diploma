#include <math.h>
#include <stdlib.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "transformer.h"
#include "solver.h"
#include "matrix.h"
#include "test.h"

const std::filesystem::path kPath = "../data.txt";

int main(int argc, char* argv[]) {
    if (argc > 1 && strcmp(argv[1], "--test") == 0) {
        TestAll();
        return 0;
    } else if (argc > 1 && strcmp(argv[1], "--ilp") == 0) {
        TestAdaptive();
        return 0;
    }
    // Matrix<double> RawC = buildC(t0, t1, N, A, B, H, g, x0);
    // Matrix RawC(1, N);
    // for (int i = 0; i < N; i++)
    //     RawC(0, i) = 1;
    // Matrix<double> RawD = buildD(t0, t1, N, A, B, H, g, x0);
    // Matrix<double> RawG0 = compute_g0(g, H, x0);

    Transformer transformer;
    transformer.Initialize();
    size_t number = transformer.GetNumber();
    Solver solver;  
    std::vector<double> u_vec(number);
    alglib::minlpstate state;
    alglib::minlpreport rep;
    alglib::real_1d_array x;
    solver.SolveRegular(state, rep, x, &transformer);
    std::cout << "\nx:\n";

    // for (int i = 0; i < N; i++) {
    //     U[i] = x(i);
    for (int idx = 0; idx < number; idx++) {
        u_vec[idx] = x(idx);
        // u_vec[idx] = 0;
        std::cout << u_vec[idx] << ' ';
    }

    auto matrix_c = transformer.GetC();
    auto left_t = transformer.GetFirstTimepoint();
    auto right_t = transformer.GetSecondTimepoint();
    
    std::cout << "\nTarget:\n";
    double ans = 0;
    double step = (right_t - left_t) / number;
    std::cout << "matrix::\n";
    std::cout << matrix_c << '\n';
    std::cout << "matrix:\n";
    for (int idx = 0; idx < number; idx++) {
        ans += x(idx) * step;
        // std::cout << ans << '\n';
    }
    
    std::cout << -ans << '\n';

    std::cout << "Computing:\n";

    auto final_pos = transformer.ComputePathWithGivenU(u_vec, kPath);
    std::cout << "FinalPos:\n";
    std::cout << final_pos;

    Matrix does_it = transformer.GetH() * final_pos;
    std::cout << "Nevyazka:\n";

    std::cout << does_it << transformer.Getg();
}
