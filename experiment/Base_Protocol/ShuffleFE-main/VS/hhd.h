#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector> 
#include <chrono> 
#include <cmath>
#include <cassert>
#include <algorithm>
#include <unordered_map>

#define ln(x) log(x)
#define log2(x) (log(x) / log(2))

uint64_t B, n;
double epsilon, delta;
double phi;

double sampleProb, filterProb;

std::vector<uint64_t> data;
std::vector<double> weight;

void loaddata(std::string file_name) {
    freopen(file_name.c_str(), "r", stdin);
    scanf("%llu", &n);
    data.clear();
    data.resize(n);
    weight.resize(n);
    for (auto i=0; i<n; ++i) {
        scanf("%llu %lf", &data[i], &weight[i]);
    }
}

uint64_t QuickPower(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t s = 1;
    while (b) {
        if (b & 1) {
            s = s * a % mod;
        }
        a = a * a % mod;
        b >>= 1;
    }
    return s;
}

bool IsPrime(uint64_t p) {
    uint32_t testPrime[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    for (auto testP : testPrime) {
        if (p % testP == 0 || QuickPower(testP, p-1, p) != 1) {
            return false;
        }
    }
    return true;
}

bool Comp(const std::pair<uint64_t, uint64_t> a, const std::pair<uint64_t, uint64_t> b) {
    return (a.second > b.second);
}