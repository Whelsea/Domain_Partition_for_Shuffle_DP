import argparse
import time
import numpy as np
import math
import collections
from math import log, log2, ceil, floor, sqrt
from bisect import bisect_right, bisect_left

from tqdm import tqdm


def load_data(file_name):
    global data
    file = open(file_name, 'r')
    data = []
    a = file.readlines()
    for i in a:
        data.append(int(i))


def get_node(B, l, r):
    nodes = []
    start = l
    next = l
    base = int(math.log2(B)) + 1
    level = int(math.log2(B)) + 1
    index = l
    segment = []
    while next < r:
        if index % 2 == 0:
            save_next = next
            next += pow(2, base - level)
            if next < r:
                level -= 1
                index = index // 2
            else:
                segment.append((start, save_next, level, index))
                level = int(math.log2(B)) + 1
                start = save_next + 1
                index = start
                next = start
        else:
            segment.append((start, next, level, index))
            level = int(math.log2(B)) + 1
            start = next + 1
            index = next + 1
            next += 1
    for (_, _, level, index) in segment:
        nodes.append(2**(level-1) + index -1)
        # print()
    # print(segment)
    return nodes


def pure_dp_noise():
    global B
    sensitivity = math.log2(B) + 1
    z = np.random.laplace(0, sensitivity / eps, size=2*B-1)
    return z


def approx_dp_noise():
    global B
    global eps
    global delta
    sensitivity = sqrt(math.log2(B)+1)
    eps_s = eps / sensitivity
    delta_s = delta / sensitivity
    sigma = sqrt(2*log(1.25/delta_s))/eps_s
    z = np.random.normal(0, sigma, size=2*B-1)
    return z


def print_info(file):
    file.write("epsilon:" + str(eps) + "\n")
    file.write("delta:" + str(delta) + "\n")
    file.write("domain size:" + str(B) + "\n")

    file.write("pureDP Linf error:" + str(error1_5) + "\n")
    file.write("pureDP 50\% error:" + str(error1_1) + "\n")
    file.write("pureDP 90\% error:" + str(error1_2) + "\n")
    file.write("pureDP 95\% error:" + str(error1_3) + "\n")
    file.write("pureDP 99\% error:" + str(error1_4) + "\n")
    file.write("pureDP average error:" + str(error1_6) + "\n")

    file.write("approxDP Linf error:" + str(error2_5) + "\n")
    file.write("approxDP 50\% error:" + str(error2_1) + "\n")
    file.write("approxDP 90\% error:" + str(error2_2) + "\n")
    file.write("approxDP 95\% error:" + str(error2_3) + "\n")
    file.write("approxDP 99\% error:" + str(error2_4) + "\n")
    file.write("approxDP average error:" + str(error2_6) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='optimal small domain range counting for shuffle model')
    parser.add_argument('--n', type=int, help='total number of user')
    parser.add_argument('--B', '--b', type=int, help='domain range, B << n')
    parser.add_argument('--dataset', type=str, default='uniform',
                        help='input data set')
    parser.add_argument('--epi', type=float, default=4, help='privacy budget')
    parser.add_argument('--rep', type=int)
    opt = parser.parse_args()
    global B
    global n
    global eps
    global delta
    global error1
    global error2
    global data
    n = opt.n
    eps = opt.epi
    delta = 1 / (n ** 2)
    error1 = []
    error2 = []
    in_file = opt.dataset

    if in_file == "uniform":
        file_name = "./uniform.txt"
    elif in_file == "AOL":
        file_name = "./AOL.txt"
    elif in_file == "zipf":
        file_name = "./zipf.txt"
    elif in_file == "gaussian":
        file_name = "./gaussian.txt"
    elif in_file == "netflix":
        file_name = "./netflix.txt"
    else:
        file_name = "./uniform.txt"
    load_data(file_name)

    if in_file == "AOL" or in_file == "netflix":
        distinct = set(data)
        domain = len(distinct)
        B = pow(2, math.ceil(math.log(domain) / math.log(2)))
        n = len(data)
        delta = 1 / (n ** 2)
    else:
        domain = opt.B
        B = opt.B
    laplace_noise = pure_dp_noise()
    gaussian_noise = approx_dp_noise()
    for l in tqdm(range(domain)):
        for r in range(l + 1, domain):
            nodes = get_node(B, l, r)
            pure_error = 0
            approx_error = 0
            for node in nodes:
                pure_error += laplace_noise[node]
                approx_error += gaussian_noise[node]
            error1.append(abs(pure_error))
            error2.append(abs(approx_error))
    error1.sort()
    error1_1 = error1[int(len(error1) * 0.5)]
    error1_2 = error1[int(len(error1) * 0.9)]
    error1_3 = error1[int(len(error1) * 0.95)]
    error1_4 = error1[int(len(error1) * 0.99)]
    error1_5 = max(error1)
    error1_6 = np.average(error1)
    # print("pure", error1_1, error1_2, error1_3, error1_4, error1_5, error1_6)
    error2.sort()
    error2_1 = error2[int(len(error2) * 0.5)]
    error2_2 = error2[int(len(error2) * 0.9)]
    error2_3 = error2[int(len(error2) * 0.95)]
    error2_4 = error2[int(len(error2) * 0.99)]
    error2_5 = max(error2)
    error2_6 = np.average(error2)
    # print("approx", error2_1, error2_2, error2_3, error2_4, error2_5, error2_6)
    out_file = open("./log/Small1D2.0/central/" + str(opt.rep) + "_" + str(opt.dataset) + "_B="+ str(B) + "_n=" + str(n) + "_eps=" + str(eps) + ".txt", 'w')
    print_info(out_file)
    out_file.close()
    print('finish')