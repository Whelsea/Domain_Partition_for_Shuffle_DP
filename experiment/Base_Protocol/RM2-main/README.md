# RM^2: Answer Counting Queries Efficiently under Shuffle Differential Privacy


### Data
For synthetic datasets:

```
cd Data
python generate.py --B 1024 --n 10000000 --dataset uniform --mode small
```
Arguments:
- n: number of participants;
- B: domain size;
- dataset: We supporst 3 dataset: uniform, zipf and gaussian;
- mode: use 'multi' and 'cube' for generating dataset for multidimensional range queries and data cube queries.

For real datasets:
Download dataset from [DATA repository](https://drive.google.com/drive/u/1/folders/11uHdvbHf9OPFhd9SYgy6ZMCymKGXze66) into this directory.


### 1D Range Queries in Small Domains
In this demo, we simulate the response messages from each user, ignore the shuffle step and calculate the range counting through all these messages. We measure the following metrics: number of messages communication cost, and error.

```
cd Small1D
python RM2.py --B 1024 --n 10000000 --epi 4 --dataset uniform --branch 2
```
Arguments:
- epi: privacy parameter $\varepsilon$ in DP;
- n: number of participants, and another privacy parameter $\delta = n^{-2}$;
- B: domain size, much smaller than n;
- dataset: input data path is '../Data/[dataset].txt'; and output directory is '../log/'. We used 5 dataset: uniform, AOL, zipf, gaussian and netflix.
- branch: branching factor, by default it is 2.

### 1D Range Queries in Large Domains

```
cd Large1D
python LargeRM2.py --epi 4 --dataset uniform
```
Arguments:
- epi: privacy parameter $\varepsilon$ in DP;
- dataset: input data path is '../Data/[dataset].txt'; and output directory is '../log/'. We used 4 dataset for large domain: uniform, Szipf, zipf and IP.
-  Other variables will be fixed, n = 1e7, B = $2^{32}$, $\delta = n^{-2}$ for synthetic dataset.

### 2D Dimensional Range Queries

```
cd Multi
python multiRM2.py --epi 4
```
Arguments:
- epi: privacy parameter $\varepsilon$ in DP.
- Other variables will be fixed, n = 1e8, B = [32,32], $\delta = n^{-2}$.

### Data Cube Queries
```
cd Small1D
python cubeRM2.py --epi 4 --dataset uniform
```
Arguments:
- epi: privacy parameter $\varepsilon$ in DP;
- dataset: input data path is '../Data/[dataset].txt'; and output directory is '../log/'. We used 2 dataset for large domain: uniform, census.
-  Other variables will be fixed, n = 1e7, B = [8,8,4,4], $\delta = n^{-2}$ for synthetic dataset.

