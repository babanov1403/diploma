#pragma once

#include <cmath>

#include "matrix.h"
#include "polynom.h"

constexpr size_t kRowPolySize = 10;
constexpr size_t kNodesCount = 15;

PolynomMatrix GetExponentPolynom(const Matrix& matrix);
double ComputeMoment(int degree, double left, double right);
std::vector<double> MakeNodes(int number, double left, double right);
