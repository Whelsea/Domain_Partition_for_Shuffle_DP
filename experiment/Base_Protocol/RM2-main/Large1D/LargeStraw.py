import argparse
import collections
import os
import time
from math import log, log2, ceil, floor
from bisect import bisect_right, bisect_left

import sys
import numpy as np
from tqdm import tqdm
import multiprocessing


def load_data(filename):
    global data
    file = open(filename, 'r')
    data = []
    a = file.readlines()
    c = np.random.randint(0, B)
    m = np.random.randint(0, B)
    while m % 2 == 0:
        m = np.random.randint(0, B)
    for i in a:
        if in_file == "Szipf":
            data.append(int(((int(i) + c) * m) % B))
        else:
            data.append(int(i))

def pre_process():
    global true_frequency
    global data
    global B
    global levelq
    global b
    global bertrand_primes
    global s
    global t
    true_frequency = np.zeros(B)
    levelq = []
    for i in range(int(log2(B)) + 1):
        if pow(2, i) < b:
            levelq.append(0)
        else:
            levelq.append(bertrand_primes[bisect_right(bertrand_primes, pow(2, i))])
    for i in range(s, int(t) + 1):
        messages[i] = []


def get_mu():
    global n
    global B
    global eps
    file = open("../mu_large.txt", 'r')
    res = 0
    a = file.readlines()
    check = str(int(n)) + " " + str(int(B)) + " " + str(int(eps))
    for i in a:
        d = i.strip().split(":")
        if d[0] == check:
            res = d[1]
            break
    return float(res)


def sub_process_randomizer(i, local_data):
    global messages
    global s
    global t
    global msg_num
    np.random.seed()
    msg = {}
    for i in range(s, int(t) + 1):
        msg[i] = 0
    for d in tqdm(local_data):
        for l in range(s, int(t) + 1):
            local_msg = local_randomizer(d, l)
            msg[l] += local_msg
    for i in range(s, int(t) + 1):
        msg_num.value += msg[i]


def local_randomizer(x, l):
    global b
    global t
    global mu
    global levelq
    local_msg = 0
    domain_size = pow(2, l)
    if domain_size < b:
        sample_prob = mu / n
        local_msg += 1
        y = np.random.binomial(domain_size, sample_prob)
        local_msg += y
    else:
        local_msg += 1
        rou = mu * b / n
        fixed_msg = floor(rou)
        remaining_msg = rou - fixed_msg
        send_msg = fixed_msg + np.random.binomial(1, remaining_msg)
        local_msg += send_msg
    return local_msg


def get_node(B, l, r):
    branch = 2
    nodes = []
    start = l
    next = l + 1
    base = int(log(B, branch))
    level = int(log(B, branch))
    index = l
    segment = []
    while next <= r:
        if index % branch == 0:
            save_next = next
            next += pow(branch, base - level + 1) - pow(branch, base - level)
            if next <= r:
                level -= 1
                index = index // branch
            else:
                segment.append((start, save_next, level, index))
                level = int(log(B, branch))
                start = save_next
                index = start
                next = start + 1
        else:
            segment.append((start, next, level, index))
            level = int(log(B, branch))
            start = next
            index = next
            next += 1
    for (i, j, level, index) in segment:
        nodes.append(int(((branch * pow(branch, level-1) - 1) / (branch - 1)) + index))
    return segment


def quick_power(x, y, mod):
    res = 1
    while y:
        if y & 1:
            res = res * x % mod
        x = x * x % mod
        y >>= 1
    return res


def range_query(l, h):
    global data
    global mu
    global levelq
    global t
    global b
    error_i = 0
    sample_prob = mu / n
    segments = get_node(B, min(l, h), max(l, h))
    noise = {}
    for (i, j, level, index) in segments:
        if (level, index) in noise.keys():
            error_i += noise[(level, index)]
        else:
            domain_size = pow(2, level)
            if domain_size < b:
                noise1 = np.random.binomial(n, sample_prob)
                error_i = error_i + noise1 - mu
                noise[(level, index)] = noise1 - mu
            else:
                q = levelq[level]
                collision_prob = 1.0 * (q / b) * (q % b + q - b) / (1.0 * q * (q - 1))
                rou = mu * b / n
                fixed_msg = floor(rou)
                remaining_msg = rou - fixed_msg
                noise1 = np.random.binomial(n , collision_prob)
                noise2 = np.random.binomial(n * fixed_msg, 1.0 / b)
                noise3 = np.random.binomial(n, remaining_msg / b)
                res = (noise1 + noise2 + noise3 - n * rou / b - n * collision_prob)
                error_i = error_i + res
                noise[(level, index)] = res
    return error_i


def true_result(l, h):
    global data
    left = bisect_left(data, min(l, h))
    right = bisect_left(data, max(l, h))
    return right - left


def print_info(file):
    file.write("epsilon:" + str(eps) + "\n")
    file.write("delta:" + str(delta) + "\n")
    file.write("number of participants:" + str(n) + "\n")
    file.write("large domain size:" + str(B) + "\n")
    file.write("real number of message / user:" + str(msg_num.value / n) + "\n")

    file.write("Linf error:" + str(error_5) + "\n")
    file.write("50\% error:" + str(error_1) + "\n")
    file.write("90\% error:" + str(error_2) + "\n")
    file.write("95\% error:" + str(error_3) + "\n")
    file.write("99\% error:" + str(error_4) + "\n")
    file.write("average error:" + str(error_6) + "\n")


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    global messages
    global n
    global B
    global b
    global phi
    global eps
    global delta
    global pf
    global mu
    global rho
    global data
    global true_frequency
    global levelq
    global bertrand_primes
    global branch
    global error
    global msg_num
    multiprocessing.set_start_method("fork")
    parser = argparse.ArgumentParser(description='optimal small domain range counting for shuffle model')
    parser.add_argument('--dataset', type=str, default='uniform',
                        help='input data set')
    parser.add_argument('--epi', type=float, default=10, help='privacy budget')
    parser.add_argument('--rep', type=int, default=0)
    opt = parser.parse_args()
    bertrand_primes = [
        2, 3, 5, 7, 13, 23,
        43, 83, 163, 317, 631, 1259,
        2503, 5003, 9973, 19937, 39869, 79699,
        159389, 318751, 637499, 1274989, 2549951, 5099893,
        10199767, 20399531, 40799041, 81598067, 163196129, 326392249,
        652784471, 1305568919, 2611137817, 5222275627]
    branch = 2
    manager = multiprocessing.Manager()
    messages = manager.dict()
    msg_num = manager.Value(int, 0)
    data = opt.dataset
    # fixed n and B
    if data == "IP":
        B = pow(2, 32)
        n = 12564270
    else:
        B = pow(2, 30)
        n = 1e7
    eps = opt.epi
    # eps = 10
    delta = 1 / (n * n)
    s = 1
    t = log2(B)
    c = 3
    beta = 0.1
    r = t - s + 1
    b = ceil(n / pow(log2(n), c))
    delta_s = delta / r
    eps_s = eps / r
    mu = get_mu()
    # fixed
    print(mu * b / n, b)
    pre_process()
    in_file = opt.dataset
    if in_file == "uniform":
        file_name = "../Data/uniform_Large.txt"
    elif in_file == "IP":
        file_name = "../Data/ip_all.txt"
    elif in_file == "zipf":
        file_name = "../Data/zipf.txt"
    elif in_file == "Szipf":
        file_name = "../Data/zipf.txt"
    else:
        file_name = "../Data/uniform.txt"
    load_data(file_name)
    process_num = 10
    index = n // 10
    result = []
    start_time = time.time()
    manager = multiprocessing.Manager()
    # messages = manager.dict()
    for i in range(process_num):
        # Try to make  parameters locally
        if i < process_num - 1:
            left = index * i
            right = index * (i + 1)
        else:
            left = index * i
            right = n
        local_data = data[int(left):int(right)]
        result.append(multiprocessing.Process(target=sub_process_randomizer, args=(i, local_data)))
        result[i].start()
    for i in range(process_num):
        result[i].join()
    
    error = []
    data.sort()
    for i in tqdm(range(100000)):
        l = np.random.randint(0, B)
        h = np.random.randint(0, B)
        # l = 0
        # h = B-2
        while h == l:
            h = np.random.randint(0, B)
        error.append(abs(range_query(l, h)))
    global error_1
    global error_2
    global error_3
    global error_4
    global error_5
    global error_6
    error.sort()
    error_1 = error[int(len(error) * 0.5)]
    error_2 = error[int(len(error) * 0.9)]
    error_3 = error[int(len(error) * 0.95)]
    error_4 = error[int(len(error) * 0.99)]
    error_5 = max(error)
    error_6 = np.average(error)
    out_file = open("../log/"+"Large1D_StrawMan_" + str(opt.rep) + str(opt.dataset) + "_eps=" + str(eps) + ".txt", 'w')
    print_info(out_file)
    print("finish")
    out_file.close()
