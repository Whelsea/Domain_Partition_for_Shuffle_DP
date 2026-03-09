#include "hhd.h"

class HeavyHitterDetection {
public:
    HeavyHitterDetection (uint64_t number_of_participants, uint64_t field_size, \
            double _phi, double _epsilon, double _delta) {
        n = number_of_participants;
        B = field_size;
        phi = _phi;
        epsilon = _epsilon;
        delta = _delta;

        // s = (uint64_t) ceil(log2(n / log2(n)));
        s = (uint64_t) ceil( log2(n/(phi * log2(n))) );
        t = (uint64_t) ceil(log2(B));
        s = std::min(s, t-1);
        b = (uint64_t) ceil(n / pow(log2(n), 2));
        pf = std::min(8.0 * (t-s) / (phi * n) * log((t-s) / (phi*0.1)), 30.0 * (t-s) / (phi * n) * log(phi * B));

        std::cerr << "filtering level " << s << " - " << t << std::endl;
        std::cerr << "pf = " << pf << std::endl;

        // Frequency Estimation Protocol
        // B = 2^i ; n = n / 2(t-s) ; (epsilon, delta/2)
        uint64_t fen = n / (2*(t-s));
        hhthreshold = (uint64_t) ceil(phi * n);
        Delta = (uint64_t) ceil(pf * phi * fen);

        mu = SearchOptMu(epsilon, delta/2, fen);

        levelq.resize(t+1);
        for (auto i=s; i<t; ++i) {
            for (levelq[i] = (1ULL<<i) + 1; !IsPrime(levelq[i]); ++levelq[i]);
            std::cerr << "level " << i << " big prime is " << levelq[i] << std::endl;
        }

        sample_message = mu * b / fen;
        std::cerr << mu << ' ' << b << ' ' << fen << std::endl;
        fixed_number_message = (uint64_t) floor(sample_message);
        remaining_message = sample_message - fixed_number_message;
        
        expected_message = sample_message * pf;

        levelMessages.resize(t+1);

        std::cerr << pf * phi * n / (t - s) << ' ' << Delta << ' ' << mu << std::endl;
    }

    void LocalRandomizer(uint64_t value) {
        std::clock_t start_time = clock();

        std::random_device rd; // private randomness
        std::mt19937 generator(rd());
        std::bernoulli_distribution fp(pf);

        // selected layer
        std::uniform_int_distribution<uint64_t> level(s, t-1);
        uint64_t level_id = level(generator);
        uint64_t q = levelq[level_id];

        // std::cerr << value << ' ' << (value >> (t - level_id)) << std::endl;

        value = value >> (t - level_id);

        // FE1 protocol
        std::vector<uint64_t> message;
        std::uniform_int_distribution<uint64_t> unifu(1, q-1);
        std::uniform_int_distribution<uint64_t> unifv(1, q);
        std::uniform_int_distribution<uint64_t> unifb(0, b-1);

        // real response
        uint64_t u, v, w;
        u = unifu(generator);
        v = unifv(generator);
        // w = ((u * value + v) % q) % b;
        w = (QuickMult(u, value, q) + v) % q % b;
        message = {u, v, w};
        if (fp(generator) == true) {
            levelMessages[level_id].push_back(message);
        }
        // dummy response
        uint64_t send_messages = fixed_number_message;
        std::bernoulli_distribution ber(remaining_message);
        if (ber(generator) == true) {
            send_messages += 1;
        }

        for (auto i=0; i<send_messages; ++i) {
            u = unifu(generator);
            v = unifv(generator);
            w = unifb(generator);
            message = {u, v, w};
            if (fp(generator) == true) {
                levelMessages[level_id].push_back(message);
            }
        }
        std::clock_t end_time = clock();
        user_time += 1.0 * (end_time - start_time) / CLOCKS_PER_SEC;
    }

    uint64_t Counter(uint64_t level_id, uint64_t query_id) {
        uint64_t counter = 0;
        uint64_t q = levelq[level_id];
        for (auto message : levelMessages[level_id]) {
            uint64_t u = message[0], v = message[1], w = message[2];
            // if ((u * query_id + v) % q % b == w) {
            if ((QuickMult(u, query_id, q) + v) % q % b == w) {
                ++ counter;
            }
        }
        return counter;
    }

    void AllCounter(uint64_t level_id) {
        uint64_t B = (1ULL << level_id);
        uint64_t q = levelq[level_id];
        counters.clear();
        counters.resize(B, 0);
        for (auto message : levelMessages[level_id]) {
            uint64_t u = message[0];
            uint64_t v = message[1];
            uint64_t w = message[2];
            uint64_t invu = QuickPower(u, q-2, q);
            // uint64_t start_id = invu * (w - v + q) % q;
            uint64_t start_id = QuickMult(invu, w-v+q, q);
            // uint64_t adding = invu * b % q;
            uint64_t adding = QuickMult(invu, b, q);
            for (uint64_t j=0, id = start_id; j<=q/b; ++j) {
                if (0 <= id && id < B) {
                    counters[id] += 1;
                }
                id += adding;
                if (id >= q) {
                    id -= q;
                }
            }
        }
    }

    void Analyzer() {
        std::clock_t start_time = clock();
        std::vector<uint64_t> potids, qryids;

        AllCounter(s);
        for (auto i=0; i<(1ULL << s); ++i) {
            if (counters[i] >= Delta) {
                potids.push_back(i);
            }
        }

        for (auto level=s+1; level<t; ++level) {
            std::cerr << "level " << level-1 << " has " << potids.size() << " candidates and " << levelMessages[level-1].size() << " messages \n";
            qryids.clear();
            for (auto i : potids) {
                if (Counter(level, i+i) >= Delta) {
                    qryids.push_back(i+i);
                }
                if (Counter(level, i+i+1) >= Delta) {
                    qryids.push_back(i+i+1);
                }
            }
            potids = qryids;
        }

        for (auto i : potids) {
            hhcandidates.push_back(i+i);
            hhcandidates.push_back(i+i+1);
        }

        std::clock_t end_time = clock();
        analyzer_time += 1.0 * (end_time - start_time) / CLOCKS_PER_SEC;
    }

    void Checker(std::vector<uint64_t> data) {
        for (auto i : data) {
            if (fevec.find(i) == fevec.end()) {
                fevec[i] = 1;
            } else {
                fevec[i] = fevec[i] + 1;
            }
        }
        for (auto i : fevec) {
            if (i.second >= hhthreshold) {
                realhh.push_back(i.first);
            }
        }
    }

    void PrintInfo() {
        std::cout << "number of participants = " << n << std::endl;
        std::cout << "field size = " << B << std::endl;
        std::cout << "candidate threshold = " << phi << std::endl;
        std::cout << std::endl;

        std::cout << "filter messages = " << expected_message << std::endl;
        std::cout << "local randomizer time cost = " << user_time << std::endl;
        std::cout << "analyzer time cost = " << analyzer_time << std::endl;
        std::cout << std::endl;

        std::vector<uint64_t> bothhh;
        for (auto i : hhcandidates) {
            if (fevec.find(i) != fevec.end() && fevec[i] >= hhthreshold) {
                bothhh.push_back(i);
            }
        }
        std::cout << "recall rate = " << (realhh.size() == 0 ? 100.0 : 100.0 * bothhh.size() / realhh.size()) << std::endl;
        std::cout << "precision rate = " << (hhcandidates.size() == 0 ? 100.0 : 100.0 * bothhh.size() / hhcandidates.size()) << std::endl;
        std::cout << std::endl;

        std::cout << "real heavy hitter :" << std::endl;
        for (auto i : realhh) {
            std::cout << i << ',';
        }
        std::cout << std::endl << std::endl;

        std::cout << "detected heavy hitter :" << std::endl;
        for (auto i : hhcandidates) {
            std::cout << i << ',';
        }

        std::cout << std::endl;
    }

private:
    uint64_t n, B, Delta, hhthreshold;
    double epsilon, delta, phi, pf, mu;
    uint64_t b, s, t;
    double sample_message, remaining_message;
    uint64_t fixed_number_message;

    std::vector<uint64_t> levelq;
    std::vector<std::vector<std::vector<uint64_t>>> levelMessages;
    std::vector<uint64_t> counters, realhh, hhcandidates;
    std::unordered_map<uint64_t, uint64_t> fevec;
    double expected_message, user_time, analyzer_time;

    // finding optimal mu
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
        // return ri * n;
        return 32 * log(2 / delta) / (epsilon * epsilon);
    }
};

int main() {
    loaddata("../data/aol01,n=10000000,B=2^48.txt");

    phi = 0.005;
    epsilon = 1;
    delta = 1.0 / n / n;
    
    std::cerr << n << ' ' << B << ' ' << data[0] << ' ' << data[1] << std::endl;

    HeavyHitterDetection hhd(n, B, phi, epsilon, delta);

    for (auto i=0; i<data.size(); ++i) {
        hhd.LocalRandomizer(data[i]);
        if (i % (data.size() / 100) == 0) {
            std::cerr << "local randomizer percentage " << 100.0 * i / data.size() << std::endl;
        }
    }
    hhd.Analyzer();
    hhd.Checker(data);
    freopen("../result/hhd.out", "w", stdout);
    hhd.PrintInfo();

    return 0;
}