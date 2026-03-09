#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector> 
#include <chrono> 
#include <cmath>
#include <cassert>
#include <algorithm>

#define ln(x) log(x)
#define log2(x) (log(x) / log(2))

using sysclock_t = std::chrono::system_clock;

std::string CurrentDate()
{
    std::time_t now = sysclock_t::to_time_t(sysclock_t::now());

    char buf[16] = { 0 };
    std::strftime(buf, sizeof(buf), "%m-%d-%H-%M-%S", std::localtime(&now));
    
    return std::string(buf);
}