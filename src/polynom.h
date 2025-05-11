#pragma once
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "utils.h"

double ComputePower(double val, int power);

class Polynom {
public:
    Polynom(int size);
    Polynom(const std::vector<double> &ceofs);
    Polynom(const std::initializer_list<double> &list);
    Polynom(const Polynom &p);
    Polynom(Polynom &&p);

    double &operator[](int idx);
    double operator[](int idx) const;

    Polynom &operator*(double x);

    double GetValue(double val);

    friend std::ostream &operator<<(std::ostream &, const Polynom &);

private:
    size_t size_;
    std::vector<double> coefs_;
};