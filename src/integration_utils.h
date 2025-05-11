#pragma once

#include <cmath>

#include "matrix.h"
#include "polynom.h"

constexpr size_t kRowPolySize = 25;
constexpr size_t kNodesCount = 25;

PolynomMatrix GetExponentPolynom(const Matrix& matrix);
double ComputeMoment(int degree, double left, double right);
std::vector<double> MakeNodes(int number, double left, double right);
