#include "hhd.h"

    long double epow;
    std::vector<long double> prob, accprob, tp, tp2;
    bool Checker(long double para, long double delta, uint64_t n) {
        long double C = 1.0;
        tp[0] = tp2[0] = 1.0;
        for (auto i=1; i<=n; ++i) {
            tp[i] = tp[i-1] * para;
            tp2[i] = tp2[i-1] * (1 - para);
        }
        for (auto i=0; i<=n; ++i) {
            prob[i] = C * tp2[n-i]; // C(n, i) * p^i * (1-p)^(n-i)
            C = C * (n-i) * para / (i+1);
        }
        accprob [n] = prob [n];
        for (int i=n-1; i>=0; --i) {
            accprob[i] = accprob[i+1] + prob[i];
        }
        long double pro = 0.0;
        for (int x2=0; x2<=n; ++x2) {
            int x1 = (int)ceil(epow * x2 - 1);
            if (x1 < 0) {
                x1 = 0;
            }
            if (x1 >= n) {
                break;
            }
            pro += prob[x2] * accprob[x1];
        }
        return (pro <= delta);
    }
    double SearchOptMu(double epsilon, double delta, uint64_t n) {
        epow = exp(epsilon);

        prob.resize(n+1);
        accprob.resize(n+1);
        tp.resize(n+1);
        tp2.resize(n+1);

        double le, ri, mi;
        le = 0.0; ri = 32.0 * log(2/delta) / epsilon / epsilon / n;
        if (ri * n >= .5e4) {
            le = ri;
        }
        while (le + 0.1 / n < ri) {
            mi = (le + ri) * .5;
            if (Checker((long double)mi, (long double)delta, n)) {
                ri = mi;
            } else {
                le = mi;
            }
        }
        return ri * n;
    }

class LargeDomainFrequencyEstimation{
public:
    LargeDomainFrequencyEstimation ();
    LargeDomainFrequencyEstimation (uint64_t number_of_parties, uint64_t field_size, double mu, double util_para = 1.0) {
        n = number_of_parties;
        B = field_size;
        c = util_para;
        b = n / pow(log(n), c);
        for (q = std::max(B, b) + 1; !IsPrime(q); ++q);
        _mu = mu;
        sample_prob = mu * (1.0 * b / n);
        send_fixed_messages = (uint64_t) floor(sample_prob);
        remaining_prob = sample_prob - send_fixed_messages;
        collision_prob = 1.0 * (q / b) * (q % b + q - b) / (1.0 * q * (q - 1));
        user_time = analyzer_time = 0.0;
    }

    void LocalRandomizer(uint64_t value, double weight) {
        std::clock_t start_time = clock();
        std::random_device rd; // private randomness
        std::mt19937 generator(rd());

        std::uniform_int_distribution<uint64_t> unifu(1, q-1);
        std::uniform_int_distribution<uint64_t> unifv(1, q);
        std::uniform_int_distribution<uint64_t> unifb(0, b-1);

        // real response
        std::vector<uint64_t> message;
        uint64_t u, v, w;

        std::bernoulli_distribution b1(weight);
        if (b1(generator) == false) {
            value = 0;
        }
        
        u = unifu(generator);
        v = unifv(generator);
        w = ((u * value + v) % q) % b;
        message = {u, v, w};
        messages.push_back(message);

        // dummy response
        
        uint64_t send_messages = send_fixed_messages;
        std::bernoulli_distribution ber(remaining_prob);
        if (ber(generator) == true) {
            send_messages += 1;
        }

        for (auto i=0; i<send_fixed_messages; ++i) {
            u = unifu(generator);
            v = unifv(generator);
            w = unifb(generator);
            message = {u, v, w};
            messages.push_back(message);
        }
        std::clock_t end_time = clock();
        user_time += 1.0 * (end_time - start_time) / CLOCKS_PER_SEC;
    }

    double Analyzer(uint64_t query_id) {
        uint64_t counter = 0;
        for (auto i=0; i<messages.size(); ++i) {
            uint64_t u = messages[i][0], v = messages[i][1], w = messages[i][2];
            if ((u * query_id + v) % q % b == w) {
                ++ counter;
            }
        }
        // std::cout << query_id << ',' << counter << ' ' << collision_prob << std::endl;
        return (counter - n * sample_prob / b - n * collision_prob) / (1 - collision_prob);
    }

    void Analyzers(std::vector<uint64_t> qids) {
        std::cerr << "analyzing ..." << ' ' << n << ' ' << B << ' ' << qids.size() << std::endl;
        std::clock_t start_time = clock();
        freqvec.resize(B+1, 0);
        for (auto i=0; i<qids.size(); ++i) {
            // std::cerr << i << ' ' << qids[i] << ' ' << B << ' ' << (qids[i] < 1 || qids[i] >= B) << std::endl;
            if (qids[i] < 1 || qids[i] >= B) {
                continue;
            }
            freqvec[qids[i]] = Analyzer(qids[i]);
            // std::cerr << i << ' ' << qids[i] << ' ' << freqvec[qids[i]] << std::endl;
        }
        std::clock_t end_time = clock();
        std::cerr << "analyze finished" << std::endl;
        analyzer_time += 1.0 * (end_time - start_time) / CLOCKS_PER_SEC;
    }

    void CheckError(std::vector<uint64_t> data, std::vector<double> weight, std::vector<uint64_t> hhc) {
        std::cerr << "checking error ..." << std::endl;
        realvec.clear();
        realvec.resize(B + 1);
        for (auto i=0; i<data.size(); ++i) {
            realvec[data[i]] += weight[i];
        }
        double heavy, light, zero, error, zeromax;
        heavy = 0; light = 0; zero = 0; zeromax = -1;

        std::unordered_map<uint64_t, bool> isinhhc;
        for (auto i : hhc) {
            isinhhc[i] = true;
        }

        for (auto i=1; i<B; ++i) {
            if (freqvec[i] < 700) {
                freqvec[i] = 0;
            }
            
            error = fabs(realvec[i] - freqvec[i]);
            l1Error += error;
            l2Error += error * error;
            looError = std::max(looError, error);
            
            if (isinhhc.find(i) != isinhhc.end()) {
                heavy += error * error;
            } else {
                light += error * error;
            }
            if (realvec[i] == 0) {
                zero += error * error;
                zeromax = std::max(zeromax, error);
            }
        }
        std::cerr << "l2 = " << l2Error << std::endl;
        std::cerr << "heavy = " << heavy << std::endl;
        std::cerr << "light = " << light << std::endl;
        std::cerr << "zero = " << zero << " , " << zeromax << std::endl;


    }

    void PrintInfo(std::string filename) {
        freopen(filename.c_str(), "w", stdout);

        std::cout << "epsilon = " << epsilon << std::endl;
        std::cout << "delta = " << delta << std::endl;
        std::cout << "number of participants = " << n << std::endl;
        std::cout << "field size = " << B << std::endl;

        std::cout << "utility parameter = " << c << std::endl;
        std::cout << "modular size = " << b << std::endl;
        std::cout << "big prime = " << q << std::endl;
        std::cout << "mu = " << _mu << std::endl;

        std::cout << "collision probability = " << collision_prob << std::endl;

        std::cout << std::endl;

        std::cout << "expected number of message = " << 1 + sample_prob << std::endl;
        comm_cost = (sample_prob + 1) * (ceil(log2(q)) * 2 + ceil(log2(b))) / 8;
        std::cout << "expected communication cost / user = " << comm_cost << " bytes" << std::endl;
        std::cout << "expected total communication cost = " << n * comm_cost << " bytes" << std::endl;
        std::cout << "real total communication cost = " << messages.size() * (ceil(log2(q)) * 2 + ceil(log2(b))) / 8 << " bytes" << std::endl;
        std::cout << std::endl;

        std::cout << "L1 error = " << l1Error << std::endl;
        std::cout << "L2 error = " << l2Error << std::endl;
        std::cout << "Linf error = " << looError << std::endl;
        std::cout << std::endl;

        std::cout << "local randomizer time cost = " << user_time << std::endl;
        std::cout << "analyzer time cost = " << analyzer_time << std::endl;
    }

private:
    uint64_t n, B, b, q, send_fixed_messages;
    double c, _mu, sample_prob, collision_prob, remaining_prob;
    double comm_cost, l1Error, l2Error, looError, l50Error, l90Error, l95Error, l99Error;
    double user_time, analyzer_time;

    std::vector<std::vector<uint64_t>> messages;
    std::vector<double> freqvec, realvec, errors;
};

std::vector<uint64_t> loadhhcfile(std::string filename) {
    freopen(filename.c_str(), "r", stdin);
    std::vector<uint64_t> hhc;
    uint64_t id;
    while (scanf("%lld,", &id) != EOF) {
        hhc.push_back(id);
    }
    return hhc;
}

int main() {

    std::string filename = "aol";

    std::string cinfilename = "../data/" + filename + ".txt";
    loaddata(cinfilename.c_str());

    B = 1ULL << 30;

    if (filename == "aol") {
        for (auto i=0; i<data.size(); ++i) {
            data[i] >>= 18;
        }
    }

    epsilon = 1;
    delta = 1.0 / n / n;
    phi = 0.0007;

    std::cerr << n << ' '<< B << ' ' << data[0] << ' ' << data[1] << std::endl;

    std::string hhcfilename = "hhc/" + filename + ",phi=" + std::to_string(phi) + ".txt";
    std::vector<uint64_t> hhc = loadhhcfile(hhcfilename);
    std::cerr << "load candidate size " << hhc.size() << ' ' << hhc[0] << ' ' << hhc[1] << std::endl;
    double mu = SearchOptMu(epsilon * 0.5, delta * 0.5, n);

    LargeDomainFrequencyEstimation ldfe(n, B, mu, 3.0);

    for (auto i=0; i<data.size(); ++i) {
        ldfe.LocalRandomizer(data[i], weight[i]);
        if (i % (data.size() / 100) == 0) {
            std::cerr << "local randomizer percentage " << 100.0 * i / data.size() << std::endl;
        }
    }
    ldfe.Analyzers(hhc);
    ldfe.CheckError(data, weight, hhc);
    std::string outfilename = "../result/vs," + filename + ",phi=" + std::to_string(phi) + ".out";
    ldfe.PrintInfo(outfilename);

    return 0;
}