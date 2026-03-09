#include "fe.h"

class LargeDomainFrequencyEstimation{
public:
    LargeDomainFrequencyEstimation ();
    LargeDomainFrequencyEstimation (uint64_t number_of_parties, uint64_t field_size, double mu, double util_para = 1.0) {
        n = number_of_parties;
        B = field_size;
        c = util_para;
        b = n / pow(log(n), c);
        // b = std::max(1.0, epsilon * epsilon * n / std::pow(std::log(n), c));
        for (q = std::max(B, b) + 1; !IsPrime(q); ++q);
        _mu = mu;
        sample_prob = mu * (1.0 * b / n);
        send_fixed_messages = (uint64_t) floor(sample_prob);
        remaining_prob = sample_prob - send_fixed_messages;
        collision_prob = 1.0 * (q / b) * (q % b + q - b) / (1.0 * q * (q - 1));
        user_time = analyzer_time = 0.0;
    }

    void LocalRandomizer(uint64_t value) {
        std::clock_t start_time = clock();
        std::random_device rd; // private randomness
        std::mt19937 generator(rd());

        std::uniform_int_distribution<uint64_t> unifu(1, q-1);
        std::uniform_int_distribution<uint64_t> unifv(1, q);
        std::uniform_int_distribution<uint64_t> unifb(0, b-1);

        // real response
        std::vector<uint64_t> message;
        uint64_t u, v, w;
        
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

        for (auto i=0; i<send_messages; ++i) {
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

    void Analyzers() {
        std::clock_t start_time = clock();
        freqvec.resize(B+1);
        for (auto i=1; i<=B; ++i) {
            freqvec[i] = Analyzer(i);
        }
        std::clock_t end_time = clock();
        analyzer_time += 1.0 * (end_time - start_time) / CLOCKS_PER_SEC;
    }

    void AnalyzersSpeedUp() {
        std::clock_t start_time = clock();
        freqvec.clear();
        freqvec.resize(B+1);

        // std::cerr << "total received messages " << messages.size() << std::endl;
        printf("   total received messages %d\n", messages.size());
        uint64_t bound = messages.size() / 100;
        for (auto i=0; i<messages.size(); ++i) {
            if (i % bound == 0) {
                printf("   procedures %.2lf\%\n", (i * 100.0) / messages.size());
            }
            uint64_t u = messages[i][0];
            uint64_t v = messages[i][1];
            uint64_t w = messages[i][2];
            // std::cerr << "message " << u << ' ' << v << ' ' << w << std::endl;
            uint64_t invu = QuickPower(u, q-2, q);
            uint64_t start_id = invu * (w - v + q) % q;
            uint64_t adding = invu * b % q;
            for (uint64_t j=0, id = start_id; j<=q/b; ++j) {
                if (1 <= id && id <= B) {
                    // std::cerr << "==> " << u << ' ' << v << ' ' << w << ' ' << id << std::endl;
                    freqvec[id] += 1;
                }
                id += adding;
                if (id >= q) {
                    id -= q;
                }
            }
        }
        for (auto i=1; i<=B; ++i) {
            freqvec[i] = (freqvec[i] - n * sample_prob / b - n * collision_prob) / (1 - collision_prob);
        }

        std::clock_t end_time = clock();
        analyzer_time += 1.0 * (end_time - start_time) / CLOCKS_PER_SEC;
    }

    void CheckError(std::vector<uint64_t> data) {
        realvec.clear();
        realvec.resize(B + 1);
        for (auto i=0; i<data.size(); ++i) {
            realvec[data[i]] += 1;
        }
        double error;
        for (auto i=1; i<=B; ++i) {
            if (realvec[i] != 0) {
                // std::cerr << i << ' ' << realvec[i] << ' ' << freqvec[i] << std::endl;
            }
            error = fabs(realvec[i] - freqvec[i]);
            l1Error += error;
            l2Error += error * error;
            looError = std::max(looError, error);
            errors.push_back(fabs(realvec[i] - freqvec[i]));
        }
        l2Error = sqrt(l2Error);
        sort(errors.begin(), errors.end());
        l99Error = errors[(uint64_t)round(0.99 * B) - 1];
        l95Error = errors[(uint64_t)round(0.95 * B) - 1];
        l90Error = errors[(uint64_t)round(0.90 * B) - 1];
        l50Error = errors[(uint64_t)round(0.50 * B) - 1];
    }

    void PrintInfo() {
        std::cout << "epsilon = " << epsilon << std::endl;
        std::cout << "delta = " << delta << std::endl;
        std::cout << "number of participants = " << n << std::endl;
        std::cout << "field size = " << B << std::endl;

        std::cout << "utility parameter = " << c << std::endl;
        std::cout << "modular size = " << b << std::endl;
        std::cout << "big prime = " << q << std::endl;
        std::cout << "mu = " << _mu << std::endl;
        std::cout << "b = " << b << std::endl;
        std::cout << "n = " << n << std::endl;

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
        std::cout << "50\% error = " << l50Error << std::endl;
        std::cout << "90\% error = " << l90Error << std::endl;
        std::cout << "95\% error = " << l95Error << std::endl;
        std::cout << "99\% error = " << l99Error << std::endl;
        std::cout << std::endl;

        std::cout << "local randomizer time cost = " << user_time << std::endl;
        std::cout << "analyzer time cost = " << analyzer_time << std::endl;
        std::cout << "single element query time cost = " << element_time << std::endl;
    }

    void SetElementQueryTime(double tim) {
        element_time = tim;
    }

private:
    uint64_t n, B, b, q, send_fixed_messages;
    double c, _mu, sample_prob, collision_prob, remaining_prob;
    double comm_cost, l1Error, l2Error, looError, l50Error, l90Error, l95Error, l99Error;
    double user_time, analyzer_time, element_time;

    std::vector<std::vector<uint64_t>> messages;
    std::vector<double> freqvec, realvec, errors;
};

double epow;
std::vector<double> prob, accprob, tp, tp2;

bool Checker(double para) {
    double C = 1.0;
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
    double pro = 0.0;
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

double Search() {
    epow = exp(epsilon);

    prob.resize(n+1);
    accprob.resize(n+1);
    tp.resize(n+1);
    tp2.resize(n+1);

    double le, ri, mi;
    le = 0.0; ri = 1000.0 / n;
    while (le + 0.1 / n < ri) {
        mi = (le + ri) * .5;
        if (Checker(mi)) {
            ri = mi;
        } else {
            le = mi;
        }
    }
    return ri * n;
    // return 32 * log(2 / delta) / (epsilon * epsilon);
}

/*
arguments: epsilon, n, B, c, filename
*/

int main(int argc,char* argv[]) {
    assert (argc == 6);

    epsilon = std::stod(argv[1]);
    n = std::stoll(argv[2]);
    B = std::atoll(argv[3]);
    double c = std::stod(argv[4]);
    delta = 1.0 / n / n;
    
    std::string filename = argv[5];
    filename = filename + "," + "n=" + std::to_string(n) + "," + "B=" + std::to_string(B);

    loaddata("../data/" + filename+ ".txt");

    filename = filename + ",C=" + std::to_string(c);

    std::string timestr = CurrentDate();

    double mu = Search();

    LargeDomainFrequencyEstimation ldfe(n, B, mu, c);
    // printf("Finished Initialiation\n");

    for (auto i=0; i<data.size(); ++i) {
        ldfe.LocalRandomizer(data[i]);
    }
    // printf("Finished Local Randomizers\n");

    ldfe.AnalyzersSpeedUp();
    // printf("Finished Analyzers\n");
    ldfe.CheckError(data);
    // printf("Finished Error Check\n");
    auto start_time = clock();
    for (auto qryid=1; qryid <= 100; ++qryid) {
        ldfe.Analyzer(qryid);
    }
    auto end_time = clock();
    ldfe.SetElementQueryTime(1.0 * (end_time - start_time) / 100.0 / CLOCKS_PER_SEC);
    std::string resultname = "../result/fe," + filename + "," + timestr + ".out";
    freopen(resultname.c_str(), "w", stdout);
    ldfe.PrintInfo();

    return 0;
}
