#include "test.h"

class TestNotPassedException : public std::runtime_error {
public:
    explicit TestNotPassedException(const char *what)
        : std::runtime_error(what) {}

    explicit TestNotPassedException(const std::string &what)
        : std::runtime_error(what.c_str()) {}
};

#define REQUIRE_EQUAL(first, second)                                               \
    do {                                                                           \
        auto firstValue = (first);                                                 \
        auto secondValue = (second);                                               \
        if (!(firstValue == secondValue)) {                                        \
            std::ostringstream oss;                                                \
            oss << "Require equal failed: " << #first << " != " << #second << " (" \
                    << firstValue << " != " << secondValue << ")\n";               \
            throw TestNotPassedException(oss.str());                               \
        }                                                                                \
    } while (false)

void TestAll() {
    {
        std::cout << "starting testing linear system solve...\n";
        TestLinearSystemSolve();
        std::cout << "linear system solve passed!\n";
    }

    {
        std::cout << "starting testing inversion...\n";
        TestInversion();
        std::cout << "inversion passed!\n";
    }

    {
        std::cout << "starting testing submatrix...\n";
        TestSubMatrix();
        std::cout << "submatrix passed!\n";
    }

    {
        std::cout << "starting testing stack...\n";
        TestStack();
        std::cout << "stack passed!\n";
    }
}

void TestLinearSystemSolve() {
    Matrix A = {{2, -1, 3, 2},
                {2, 3, 3, 2},
                    {3, -1, -1, 2},
                {3, -1, 3, -1}};
    Matrix b = {-2, 2, 2, -5};

    auto result = Matrix::SolveLinearSystem(A, b.GetTransposed());
    auto origin_result = Matrix{0, 1, -1, 1};
    REQUIRE_EQUAL(result.GetTransposed(), origin_result);
}

void TestInversion() {
    Matrix A = {{1, 0, -1},
                {-3, 2, 0},
                {1, -1, 1}};
    auto inversed = A.GetInversed();
    REQUIRE_EQUAL(A * inversed, Matrix(A.GetRows()));
}

void TestSubMatrix() {
    Matrix A = {{2, -1, 3, 2},
                {2, 3, 3, 2},
                {3, -1, -1, 2},
                {3, -1, 3, -1}};
    auto res = Matrix({{3, 3}, {-1, -1}});
    REQUIRE_EQUAL(A.GetSubMatrix(std::vector{1, 2}, {1, 2}), res);
    res = Matrix({-1, 3, 2});
    REQUIRE_EQUAL(A.GetSubMatrix(std::vector{0}, {1, 2, 3}), res);
    REQUIRE_EQUAL(A.GetSubMatrix(std::unordered_set{0, 1, 2, 3}, {0, 1, 2, 3}), A);
}

void TestStack() {
    Matrix first = {1, 2, 3, 4};
    Matrix second = {5, 6, 7, 8};
    auto res = Matrix({{1, 2, 3, 4}, {5, 6, 7, 8}});
    REQUIRE_EQUAL(Matrix::Stack(first, second), res);

    res = Matrix{1, 5};
    REQUIRE_EQUAL(Matrix::Stack(first.GetSubMatrix(std::vector{0}, {0}), second.GetSubMatrix(std::vector{0}, {0})), res.GetTransposed());
}

void TestAdaptive() {
    Solver solver;
    // // bad example
    // Matrix matrix_c = {2, 1};
    // matrix_c = matrix_c.GetTransposed();
    // Matrix matrix_a = {{1,2}, {-1, 1}};
    // Matrix matrix_lb = Matrix{2, -2}.GetTransposed();
    // Matrix matrix_ub = Matrix{6, -1}.GetTransposed();
    // Matrix matrix_ld = Matrix{1, 0}.GetTransposed();
    // Matrix matrix_ud = Matrix{3, 2}.GetTransposed();
    
    // auto sol = solver.SolveAdaptive(matrix_c, matrix_a, matrix_lb, matrix_ub, matrix_ld, matrix_ud);
    // std::cout << "solution is:\n";
    // std::cout << sol;

    // bad example
    Matrix matrix_c = {2, 3};
    matrix_c = matrix_c.GetTransposed();
    Matrix matrix_a = {{2,5}, {1, 1}};
    Matrix matrix_lb = Matrix{10, 3}.GetTransposed();
    Matrix matrix_ub = Matrix{10, 3}.GetTransposed();
    Matrix matrix_ld = Matrix{0, 0}.GetTransposed();
    Matrix matrix_ud = Matrix{10, 10}.GetTransposed();
    
    auto sol = solver.SolveAdaptive(matrix_c, matrix_a, matrix_lb, matrix_ub, matrix_ld, matrix_ud);
    std::cout << "solution is:\n";
    std::cout << sol;
}