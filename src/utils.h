#pragma once

#include <algorithm>
#include <cmath>
#include <set>
#include <unordered_set>
#include <vector>

constexpr double kTolerance = 1e-9;

double ComputePower(double, int);
bool EqualNotStrict(double, double);

template <class T>
std::unordered_set<T> Perece4eniye(const std::unordered_set<T>&, const std::unordered_set<T>&);

template <class T>
std::unordered_set<T> Obyedeneniye(const std::unordered_set<T>&, const std::unordered_set<T>&);
