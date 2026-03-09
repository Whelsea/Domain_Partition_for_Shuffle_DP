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

void loaddata(std::string file_name) {
    freopen(file_name.c_str(), "r", stdin);
    scanf("%llu", &n);
    scanf("%llu", &B);
    data.clear();
    data.resize(n);
    for (auto i=0; i<n; ++i) {
        scanf("%llu", &data[i]);
    }
}

uint64_t QuickMult(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t s = 0;
    while (b) {
        if (b & 1) {
            s = (s + a) % mod;
        }
        a = (a + a) % mod;
        b >>= 1;
    }
    return s;
}

uint64_t QuickPower(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t s = 1;
    while (b) {
        if (b & 1) {
            s = QuickMult(s, a, mod);
        }
        a = QuickMult(a, a, mod);
        b >>= 1;
    }
    return s;
}

bool IsPrime(uint64_t p) {
    uint64_t testPrime[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    for (uint64_t testP : testPrime) {
        if (p % testP == 0 || QuickPower(testP, p-1, p) != 1) {
            return false;
        }
    }
    return true;
}
