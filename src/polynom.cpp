#include "polynom.h"

Polynom::Polynom(int size) : size_(size) {
    coefs_ = std::vector<double>(size_);
}
Polynom::Polynom(const std::vector<double>& coefs) : coefs_(coefs) {
}

Polynom::Polynom(const std::initializer_list<double>& list) {
    coefs_ = list;
    size_ = list.size();
}

Polynom::Polynom(const Polynom& poly) {
    coefs_ = poly.coefs_;
    size_ = poly.size_;
}

Polynom::Polynom(Polynom&& poly) {
    coefs_ = std::move(poly.coefs_);
    size_ = poly.size_;
    poly.size_ = 0;
}

double& Polynom::operator[](int idx) {
    return coefs_[idx];
}

double Polynom::operator[](int idx) const {
    return coefs_[idx];
}

Polynom& Polynom::operator*(double val) {
    for (auto& el : coefs_) {
        el *= val;
    }
    return *this;
}

double Polynom::GetValue(double val) {
    double result = 0;
    for (int idx = size_ - 1; idx >= 0; --idx) {
        result += coefs_[idx] * ComputePower(val, idx);
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, const Polynom& poly) {
    if (poly.size_ == 0) {
        os << "Empty poly";
        return os;
    }
    for (int idx = poly.size_ - 1; idx >= 0; idx--) {
        std::string prefix;
        std::string suffix;
        if (poly[idx] >= 0 && idx != poly.size_ - 1) {
            prefix += '+';
        } else if (idx != poly.size_ - 1) {
            prefix += '-';
        }
        os << prefix;
        os << poly[idx];
        if (idx != 0) {
            suffix += "x^" + std::to_string(idx);
        }
        os << suffix;
    }
    return os;
}
