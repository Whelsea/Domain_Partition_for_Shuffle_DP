import time
import numpy as np
import math
import collections

from tqdm import tqdm


if __name__ == '__main__':
    Bs = [512, 1024, 2048]
    ns = [1000000, 10000000, 100000000]

    
    eps = [2 ,4, 8]
    # calculate msg number for different epi, n and B
    for B in Bs:
        for n in ns:
            for e in eps:
                delta = 1 / (n * n)
                tau = math.ceil(math.log2(n))
                k = math.ceil(math.log2(2 * B))
                rho = math.ceil((36.0 * k * k * math.log(math.e * k / (delta * e))) / (e * e))
                msg = rho + k
                print(B, n, e, msg)
