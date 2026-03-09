import argparse

import numpy as np
from math import log2, pow, log, floor

import collections
from itertools import permutations, combinations
from tqdm import tqdm


def find_opt_cube(epi, d, L, attri, eps, delta, n):
    l = 0
    total = int(pow(2, d))
    delta_s = delta / 16
    eps_s = eps / 16
    mu_1 = 32 * log(2 / delta_s) / (eps_s * eps_s)
    sample_prob = mu_1 / n
    r = n * sample_prob * (1 - sample_prob)
    save = None
    while abs(r - l) > 1 / pow(epi, 2):
        cur = (l + r) / 2
        for s in range(1, total + 1):
            flag, R = feasible(L, cur, s, attri, eps, delta, n)
            if flag:
                r = cur
                save = R
                break
        if s == total:
            l = cur
    return save


# by default, we will include the base cuboid
def feasible(L, theta, s, attri, eps, delta, n):
    temp = L.copy()

    R = {(0, 1, 2, 3)}
    cov = set()
    delta_s = delta / s
    eps_s = eps / s
    mu = 32 * log(2 / delta_s) / (eps_s * eps_s)
    sample_prob = mu / n
    var = n * sample_prob * (1 - sample_prob)
    # calculate bound
    bound = theta / var
    # coverage for base cuboid
    for cuboid in temp:
        mag = 1
        for a in (0, 1, 2, 3):
            if a not in cuboid:
                mag = mag * attri[a]
        if mag <= bound:
            cov.add(cuboid)
    temp.remove((0, 1, 2, 3))
    # greedy
    for i in range(s - 1):
        save = {}
        cur_cuboid = None
        max_cov = 0
        for cuboid in temp:
            temp_cov = set()
            for pot in temp:
                mag = 1
                if set(pot).intersection(set(cuboid)) == set(pot):
                    for a in cuboid:
                        if a not in pot:
                            mag = mag * attri[a]
                    if mag <= bound:
                        temp_cov.add(pot)
            cov_count = len(temp_cov.difference(cov))
            if cov_count >= max_cov:
                max_cov = cov_count
                cur_cuboid = cuboid
                save[cuboid] = temp_cov
        cov = cov.union(save[cur_cuboid])
        R.add(cur_cuboid)
    if cov == L:
        return True, R
    else:
        return False, None


def load_data(filename):
    global data
    file = open(filename, 'r')
    data = []
    a = file.readlines()
    for i in a:
        d = i.strip().split()
        data.append((int(d[0]), int(d[1]), int(d[2]), int(d[3])))


def get_mu():
    global n
    global eps
    file = open("../mu_cube.txt", 'r')
    res = 0
    a = file.readlines()
    check = str(int(n)) + " " + str(int(1)) + " " + str(int(eps))
    for i in a:
        d = i.strip().split(":")
        if d[0] == check:
            res = d[1]
            break
    return float(res)


def pre_process():
    global L_pre_level
    global L_level
    global L_pre
    global List_pre
    global child
    L_pre_level = {}
    L_level = {}
    child = {}
    # level for L_pre
    for cuboid in L_pre:
        if len(cuboid) in L_pre_level.keys():
            L_pre_level[len(cuboid)].append(cuboid)
        else:
            L_pre_level[len(cuboid)] = [cuboid]
    # level for L
    for cuboid in L:
        if len(cuboid) in L_level.keys():
            L_level[len(cuboid)].append(cuboid)
        else:
            L_level[len(cuboid)] = [cuboid]

    for cuboid in List_pre:
        if len(cuboid) + 1 in L_level.keys():
            save = None
            min_mag = 1000
            for From in L_level[len(cuboid) + 1]:

                if set(From).intersection(set(cuboid)) == set(cuboid):
                    mag = child[From][1]
                    for a in From:
                        if a not in cuboid:
                            mag = mag * attri[a]
                    if mag < min_mag:
                        save = From
                        min_mag = mag
            if cuboid in L_pre:
                child[cuboid] = (save, 1)
            else:
                child[cuboid] = (save, min_mag)
        else:
            child[cuboid] = (None, 1)


def local_randomizer(x, p, cells):
    global tree
    global L_pre
    global L_pre_level
    global child
    noise_msg = np.random.binomial(1, p, size=len(cells))
    noise_msg_index = np.argwhere(noise_msg == 1)
    msg = [x]
    cur_level = len([i[0] for i in np.argwhere(np.array(x) != -1)])
    # real msg
    for cuboid in L_pre:
        cub_level = len(cuboid)
        if cub_level < cur_level and cub_level in L_pre_level.keys():
            parent = [-1, -1, -1, -1]
            for j in cuboid:
                parent[j] = x[j]
            msg.append(tuple(parent))
    # noise msg
    for i in noise_msg_index:
        cell = cells[i[0]]
        msg.append(cell)
    return msg


def analyzer():
    global L_pre_level
    global L_level
    global L_pre
    global List_pre
    global child
    global messages
    global attri
    global tree
    global mu_1
    global d
    global fe_counter
    fe_counter = collections.Counter(messages)
    # debias
    for key in fe_counter:
        fe_counter[key] -= mu_1


def post_dataCube_true():
    global data
    global List_pre
    global ture_frequency
    global all_cells
    ture_frequency = collections.Counter(data)
    all_cells = []
    atr_list = [[-1], [-1], [-1], [-1]]
    for a in (0,1,2,3):
        atr_list[a] = [i for i in range(-1, attri[a], 1)]
    for x in atr_list[0]:
        for y in atr_list[1]:
            for m in atr_list[2]:
                for n in atr_list[3]:
                    all_cells.append((x, y, m, n))
    all_cells.sort(key=lambda x: list(x).count(-1))
    for cell in all_cells:
        # not a base cell
        if -1 in cell:
            cub = [i[0] for i in np.argwhere(np.array(cell) != -1)]
            From = child[tuple(cub)][0]
            dim = list(set(From).difference(set(cub)))[0]
            total = ture_frequency[cell]
            for i in range(attri[dim]):
                parent = list(cell)
                parent[dim] = i
                total += ture_frequency[tuple(parent)]
            ture_frequency[cell] = total


def post_dataCube():
    global List_pre
    global fe_counter
    global all_cells
    all_cells = []
    atr_list = [[-1], [-1], [-1], [-1]]
    for a in (0,1,2,3):
        atr_list[a] = [i for i in range(-1, attri[a], 1)]
    for x in atr_list[0]:
        for y in atr_list[1]:
            for l in atr_list[2]:
                for m in atr_list[3]:
                    all_cells.append((x, y, l, m))
    all_cells.sort(key=lambda x: list(x).count(-1))
    for cell in all_cells:
        # not a base cell
        if -1 in cell and cell not in fe_counter.keys():
            cub = [i[0] for i in np.argwhere(np.array(cell) != -1)]
            From = child[tuple(cub)][0]
            dim = list(set(From).difference(set(cub)))[0]
            total = 0
            for i in range(attri[dim]):
                parent = list(cell)
                parent[dim] = i
                total += fe_counter[tuple(parent)]
            fe_counter[cell] = total
    return fe_counter


def print_info(file):
    file.write("epsilon:" + str(eps) + "\n")
    file.write("delta:" + str(delta) + "\n")
    file.write("number of participants:" + str(n) + "\n")
    file.write("mu:" + str(mu_1) + "\n")
    file.write("dataset:" + str(in_file) + "\n")

    file.write("real number of message / user:" + str(t / n) + "\n")

    file.write("Linf error:" + str(error_5) + "\n")
    file.write("50\% error:" + str(error_1) + "\n")
    file.write("90\% error:" + str(error_2) + "\n")
    file.write("95\% error:" + str(error_3) + "\n")
    file.write("99\% error:" + str(error_4) + "\n")
    file.write("average error:" + str(error_6) + "\n")


if __name__ == '__main__':
    global tree
    global L_pre_level
    global L_level
    global L_pre
    global List_pre
    global child
    global attri
    global mu_1
    global d
    parser = argparse.ArgumentParser(description='optimal small domain range counting for shuffle model')
    parser.add_argument('--epi', type=float, default=4, help='privacy budget')
    parser.add_argument('--rep', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='uniform',
                        help='input data set')
    opt = parser.parse_args()
    # n = 2458285
    # eps = 5
    eps = opt.epi
    in_file = opt.dataset
    print("preprocess")
    if in_file == "uniform":
        file_name = "../Data/cube.txt"
        attri = [8, 8, 4, 4]
        n = 10000000
    elif in_file == "census":
        file_name = "../Data/census.txt"
        attri = [10, 7, 4, 3]
        n = 2458285
    delta = 1 / (n * n)
    L = {(0, 1, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (1,),
         (2,), (0,), (3,), ()}
    List_pre = [(0, 1, 2, 3), (1, 2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3),
                (2, 3), (1,),
                (2,), (0,), (3,), ()]
    d = 4

    L_pre = find_opt_cube(eps, d, L, attri, eps, delta, n)
    pre_process()
    print(L_pre)
    delta_s = delta / len(L_pre)
    eps_s = eps / len(L_pre)
    mu_1 = get_mu()
    print(mu_1)
    sample_prob = mu_1 / n
    tree = {}
    cells = []
    for c in L_pre:
        atr_list = [[-1], [-1], [-1], [-1]]
        for a in c:
            atr_list[a] = [i for i in range(attri[a])]
        for x in atr_list[0]:
            for y in atr_list[1]:
                for l in atr_list[2]:
                    for m in atr_list[3]:
                        tree[(x, y, l, m)] = 0
                        cells.append((x, y, l, m))
    global data
    global t
    global messages
    global error
    global ture_frequency
    global all_cells
    global fe_counter
    t = 0
    error = []
    print("preprocess")
    load_data(file_name)
    print("initialize")
    messages = []
    for dt in tqdm(data):
        msg_l = local_randomizer(dt, sample_prob, cells)
        messages += msg_l
    t = len(messages)
    analyzer()
    print("finish")
    post_dataCube_true()
    post_dataCube()
    for i in all_cells:
        error.append(abs(fe_counter[i] - ture_frequency[i]))
    error.sort()
    error_1 = error[int(len(error) * 0.5)]
    error_2 = error[int(len(error) * 0.9)]
    error_3 = error[int(len(error) * 0.95)]
    error_4 = error[int(len(error) * 0.99)]
    error_5 = max(error)
    error_6 = np.average(error)
    out_file = open(
        "../log/" + "Cube_StrawMan_" + str(opt.rep) + str(in_file)+ "_eps=" + str(eps) + ".txt", 'w')
    print_info(out_file)
    print("finish")
    out_file.close()
