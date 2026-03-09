#include "common.h"

uint64_t B, n;
uint64_t q, b;
double epsilon, delta;
double collisionProb;
double sampleProb;

std::vector<uint64_t> data;
std::vector<uint64_t> values;

// small domain
std::vector<uint64_t> messages;

// large domain
std::vector< std::vector<uint64_t> > tuples;

std::vector<uint64_t> rv, cv; //real vector & counter vector
std::vector<double> fev; // (estimated) frequency vector

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