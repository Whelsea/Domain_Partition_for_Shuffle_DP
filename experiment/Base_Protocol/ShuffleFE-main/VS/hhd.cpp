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
        t = (uint64_t) ceil(log2(B)) + 1;
        s = std::min(s, t-1);
        b = (uint64_t) ceil(n / pow(log2(n), 2));
        pf = std::min(8.0 * (t-s) / (phi * n) * log((t-s) / (phi*0.1)), 30.0 * (t-s) / (phi * n) * log(phi * B));

        std::cerr << "pf = " << pf << std::endl;
        std::cerr << "level " << s << ' ' << t << std::endl;
        // Frequency Estimation Protocol
        // B = 2^i ; n = n / 2(t-s) ; (epsilon, delta/2)
        uint64_t fen = n / (2*(t-s));
        hhthreshold = (uint64_t) ceil(phi * n);
        Delta = (uint64_t) ceil(pf * phi * fen);

        mu = SearchOptMu(epsilon, delta/2, fen);

        levelq.resize(t+1);
        for (auto i=s; i<t; ++i) {
            for (levelq[i] = (1ULL<<i) + 1; !IsPrime(levelq[i]); ++levelq[i]);
        }

        sample_message = mu * b / fen;
        std::cerr << mu << ' ' << b << ' ' << fen << std::endl;
        std::cerr << "sample_message : " << sample_message << std::endl;
        fixed_number_message = (uint64_t) floor(sample_message);
        remaining_message = sample_message - fixed_number_message;
        
        expected_message = sample_message * pf;

        levelMessages.resize(t+1);

        std::cerr << pf * phi * n / (t - s) << ' ' << Delta << ' ' << mu << std::endl;
    }

    void LocalRandomizer(uint64_t value, double weight) {
        std::clock_t start_time = clock();

        std::random_device rd; // private randomness
        std::mt19937 generator(rd());
        std::bernoulli_distribution fp(pf);

        // randomizer rounding
        std::bernoulli_distribution rr(weight);
        if (rr(generator) == false) {
            value = 0;
        }

        // selected layer
        std::uniform_int_distribution<uint64_t> level(s, t-1);
        uint64_t level_id = level(generator);
        uint64_t q = levelq[level_id];

        // std::cerr << value << ' ' << (value >> (t - level_id)) << std::endl;

        value = value >> ((t-1) - level_id);

        // FE1 protocol
        std::vector<uint64_t> message;
        std::uniform_int_distribution<uint64_t> unifu(1, q-1);
        std::uniform_int_distribution<uint64_t> unifv(1, q);
        std::uniform_int_distribution<uint64_t> unifb(0, b-1);

        // real response
        uint64_t u, v, w;
        u = unifu(generator);
        v = unifv(generator);
        w = ((u * value + v) % q) % b;
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
            if ((u * query_id + v) % q % b == w) {
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
            uint64_t start_id = invu * (w - v + q) % q;
            uint64_t adding = invu * b % q;
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
        for (auto i=0; i<10; ++i) {
            std::cerr << i << ' ' << counters[i] << ' ' << Counter(level_id, i) << std::endl;
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

        std::vector<std::pair<uint64_t, uint64_t>> sorttemp;
        if (s == t-1) {
            for (auto i : potids) {
                sorttemp.push_back(std::make_pair(i, counters[i] /*Counter(t-1, i)*/));
            }
        }

        for (auto level=s+1; level<t; ++level) {
            std::cerr << "level " << level-1 << " has " << potids.size() << " candidates and " << levelMessages[level-1].size() << " messages \n";
            qryids.clear();
            if (2 * potids.size() * b > (1ULL << level)) {
                std::cerr << "choose all counters \n";
                AllCounter(level);
                for (auto i : potids) {
                    if (counters[i+i] >= Delta) {
                        qryids.push_back(i+i);
                        if (level == t-1) {
                            sorttemp.push_back(std::make_pair(i+i, counters[i+i]));
                        }
                    }
                    if (counters[i+i+1] >= Delta) {
                        qryids.push_back(i+i+1);
                        if (level == t-1) {
                            sorttemp.push_back(std::make_pair(i+i+1, counters[i+i+1]));
                        }
                    }
                }
            } else {
                std::cerr << "choose single counter \n";
                for (auto i : potids) {
                    uint64_t count = Counter(level, i+i);
                    if (count >= Delta) {
                        qryids.push_back(i+i);
                        if (level == t-1) {
                            sorttemp.push_back(std::make_pair(i+i, count));
                        }
                    }
                    count = Counter(level, i+i+1);
                    if (Counter(level, i+i+1) >= Delta) {
                        qryids.push_back(i+i+1);
                        if (level == t-1) {
                            sorttemp.push_back(std::make_pair(i+i+1, count));
                        }
                    }
                }
            }
            potids = qryids;
        }

        std::cerr << "#candidates = " << sorttemp.size() << std::endl;

        std::sort(sorttemp.begin(), sorttemp.end(), Comp);
        std::cerr << "HH Candidates : ";
        for (auto i : sorttemp) {
            if (i.first != B-1) {
                hhcandidates.push_back(i.first);
            }
            // std::cerr << i.first << ' ';
        }
        std::cerr << std::endl;

        std::clock_t end_time = clock();
        analyzer_time += 1.0 * (end_time - start_time) / CLOCKS_PER_SEC;
    }

    void Checker(std::vector<uint64_t> data, std::vector<double> weight) {
        for (auto i=0; i<data.size(); ++i) {
            if (fevec.find(data[i]) == fevec.end()) {
                fevec[data[i]] = weight[i];
            } else {
                fevec[data[i]] = fevec[data[i]] + weight[i];
            }
        }
        for (auto i : fevec) {
            if (i.second >= hhthreshold) {
                realhh.push_back(i.first);
            }
        }
    }

    void PrintInfo(std::string filename) {
        freopen(filename.c_str(), "w", stdout);
        std::cout << "number of participants = " << n << std::endl;
        std::cout << "field size = " << B << std::endl;
        std::cout << "candidate threshold = " << phi << std::endl;
        std::cout << std::endl;

        std::cout << "filter messages : " << expected_message << std::endl;
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

        std::cout << "real heavy hitter : " << realhh.size() << std::endl;
        // for (auto i : realhh) {
        //     std::cout << i << ',';
        // }
        // std::cout << std::endl << std::endl;

        std::cout << "detected heavy hitter : " << hhcandidates.size() << std::endl;
        // for (auto i : hhcandidates) {
        //     std::cout << i << ',';
        // }

        // std::cout << std::endl;
    }

    void DetectionOutput(std::string filename) {
        freopen(filename.c_str(), "w", stdout);
        for (auto i : hhcandidates) {
            std::cout << i << ',';
        }
    }

    std::vector<uint64_t> getHHC() {
        return hhcandidates;
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
    std::unordered_map<uint64_t, double> fevec;
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
        return ri * n;
    }
};

using namespace std;

vector<uint64_t> hhc;

int main(int argc, char* argv[]) {

    std::string filename = "aol";

    std::string infilename = "../data/" + filename + ".txt";
    loaddata(infilename);

    assert(argc == 2);
    phi = stod(argv[1]);
    std::cerr << "phi = " << phi << std::endl;

    B = 1ULL<<30;
    epsilon = 1;
    delta = 1.0 / n / n;

    if (filename == "aol") {
        for (auto i=0; i<data.size(); ++i) {
            data[i] >>= 18;
        }
    }
    
    std::cerr << n << ' ' << B << ' ' << data[0] << ' ' << data[1] << std::endl;

    HeavyHitterDetection hhd(n, B, phi, epsilon / 2, delta / 2);

    for (auto i=0; i<data.size(); ++i) {
        hhd.LocalRandomizer(data[i], weight[i]);
        if (i % (data.size() / 10) == 0) {
            std::cerr << "local randomizer percentage " << 100.0 * i / data.size() << std::endl;
        }
    }
    hhd.Analyzer();

    hhd.Checker(data, weight);
    std::string outfilename = "log/" + filename + ",phi=" + std::to_string(phi) + ".txt";
    hhd.PrintInfo(outfilename);
    outfilename = "hhc/" + filename + ",phi=" + std::to_string(phi) + ".txt";
    hhd.DetectionOutput(outfilename);

    return 0;
}