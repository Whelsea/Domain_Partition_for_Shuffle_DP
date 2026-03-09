import math
import numpy as np

from decimal import Decimal
from math import log, ceil, exp, log2


def MuChecker(epsilon, delta, n, p) -> bool:
    epow = exp(epsilon)
    powp = [Decimal("1.0")] * (n + 1)  # powp[i] = p^i
    pownp = [Decimal("1.0")] * (n + 1)  # pownp[i] = (1-p)^i
    for i in range(1, n + 1):
        powp[i] = powp[i - 1] * p
        pownp[i] = pownp[i - 1] * (1 - p)

    C = Decimal("1.0")
    prob = [Decimal("1.0")] * (n + 1)  # prob[i] = Pr[Bin(n, p) = i]
    for i in range(n + 1):
        prob[i] = C * pownp[n - i]  # C(n,i) * p^i * (1-p)^(n-i)
        C = C * (n - i) * p / (i + 1)

    accprob = [Decimal("1.0")] * (n + 1)  # accprob[i] = sum_{j >= i} accprob[j]
    accprob[n] = prob[n]
    for i in range(n - 1, -1, -1):
        accprob[i] = accprob[i + 1] + prob[i]

    outp = Decimal("0.0")  # calculate sum_{X1, X2} Pr[ X1 >= e^epsilon * X2 - 1 ]
    for x2 in range(n + 1):
        x1 = int(ceil(epow * x2 - 1))
        if x1 < 0:
            x1 = 0
        elif x1 >= n:
            break
        outp += prob[x2] * accprob[x1]

    return (outp <= delta)


def get_mu():
    global n
    global B
    global eps
    file = open("./mu.txt", 'r')
    res = 0
    a = file.readlines()
    check = str(n) + " " + str(B) + " " + str(eps)
    for i in a:
        d = i.strip().split(":")
        print(d[0], d[1])
        if d[0] == check:
            print("get")
            res == d[1]
    return res


if __name__ == "__main__":
    global B
    global n
    global eps
    B = pow(2, 32)
    n = 12564270
    es = [10]
    # eps = 0.2
    # get_mu()
    # exit()
    for e in es:
        epsilon = e / log2(B)
        delta = n ** (-2) / log2(B)

        left = 0.0
        right = 32 * log(2.0 / delta) / (epsilon ** 2) / n
        Theoretical = right

        print("Theoretical Mu = ", right * n)

        while left + 1.0 / n < right:
            middle = (left + right) * .5
            if MuChecker(epsilon, delta, n, Decimal(middle)):
                right = middle
            else:
                left = middle
            # if Theoretical / right >= 8
            #     break
            print(left, right)

        file = "./mu.txt"
        out_file = open(file, 'a')
        out_file.write(str(n) + " " + str(B) + " " + str(e) + ":" + str(right * n)+ "\n")
        print("Practical Mu = ", right * n)
