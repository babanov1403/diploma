#include "utils.h"

double ComputePower(double val, int power) {
    double res = 1;
    while (power > 0) {
        if ((power & 1)) {
            res *= val;
        }
        val *= val;
        power >>= 1;
    }
    return res;
}

bool EqualNotStrict(double lhs, double rhs) {
    return std::abs(rhs - lhs) <= kTolerance;
}

template <class T>
std::unordered_set<T> Perece4eniye(const std::unordered_set<T>& lhs, const std::unordered_set<T>& rhs) {
    std::unordered_set<T> result;
    for (const auto& el : lhs) {
        if (rhs.contains(el)) {
            result.insert(el);
        }
    }
    return result;
}

template <class T>
std::unordered_set<T> Obyedeneniye(const std::unordered_set<T>& lhs, const std::unordered_set<T>& rhs) {
    std::unordered_set<T> result;
    for (const auto& el : lhs) {
        result.insert(el);
    }
    for (const auto& el : rhs) {
        result.insert(el);
    }
    return result;
}