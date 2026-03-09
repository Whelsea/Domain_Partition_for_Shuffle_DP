# SDPFE - Practical Frequency Estimation in the Shuffle Model

Qiyao Luo, Yilei Wang, and Ke Yi. 2022. Frequency Estimation in the Shuffle
Model with Almost a Single Message. In Proceedings of the 2022 ACM SIGSAC
Conference on Computer and Communications Security (CCS ’22)

### Frequency Estimation
In this demo, we simulate the response messages from each user, ignore the shuffle step and calculate the frequency estimation vector through all these messages. We measure the following metrics: number of messages communication cost, time cost, and error. we consider the domain size is no more than unsigned int for simplicity. 

```
cd FE
g++ fe.cpp -o fe -std=c++11
./fe 1 100000 16777216 3 aol01
```
Arguments (`./fe [epsilon] [n] [B] [c] [filename]`):
- epsilon: privacy parameter $\varepsilon$ in DP;
- n: number of participants, and another privacy parameter $\delta = n^{-2}$;
- B: domain size, smaller than $2^{32}$;
- c: utility parameter, hash bucket size $b = n / \log^c n$. Recommended choices: $c=1$ gets optimal error and $O(1)$ messages per user; $c=3$ gets $1+o(1)$ messages in most casses and $O(\log^2 n)$ error;
- filename: input data path is '../data/filename.txt'; and output directory is '../result'.

### Heavy Hitter Detection
```
cd HHD
g++ hhd.cpp -o hhd -std=c++11
./hhd
```
We use the $1+o(1)$ frequency estimation protocol (with $c=3$) as a component, with the domain size $B$ supports integers larger than unsigned int. The demo program chooses $n=10^7, B=2^{48}, \varepsilon=1, \delta=n^{-2}, \phi=0.005$. Parameters can also be changed in the code. The output file is '../result/hhd.out'.

### Sparse Vector Summation
```
cd VS
g++ hhd.cpp -o hhd -std=c++11
g++ fe.cpp -o fe -std=c++11
./hhd; ./fe
```
We first use HHD protocol to detect the heavy coordinates, and output the candidate indices to 'VS/hhc/'. Next, we use FE protocol to calculate the aggregation of the candidate indices. Parameter setting is $n=10^7, B=2^{30}, \varepsilon=1, \delta = n^{-2}, \phi=7\times 10^{-4}$ and the input file is 'aol' (we also provided 'zipf1', 'zipf2', 'zipf3' - Zipfian distribution file with different $\alpha$); and output directory is '../result'.