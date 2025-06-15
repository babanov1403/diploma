#include "integration_utils.h"
PolynomMatrix GetExponentPolynom(const Matrix& matrix) {
    Matrix init(matrix.GetRows());
    PolynomMatrix result(kRowPolySize);
    double curr = 1;
    result[0] = init * curr;
    for (int idx = 1; idx < kRowPolySize; idx++) {
        curr *= 1.0 / idx;
        init *= matrix;
        result[idx] = init * curr;
    }
    return result;
}
double ComputeMoment(int degree, double left, double right) {
    return ComputePower(right, degree + 1) / (degree + 1) -
           ComputePower(left, degree + 1) / (degree + 1);
}
std::vector<double> MakeNodes(int number, double left, double right) {
    std::vector<double> nodes(number + 1);
    double h = (right - left) / number;
    nodes[0] = left;
    for (int idx = 1; idx <= number; idx++) {
        nodes[idx] = nodes[idx - 1] + h;
    }
    return nodes;
}
