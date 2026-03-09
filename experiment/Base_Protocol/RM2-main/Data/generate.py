import argparse
import bisect
import math
import random

import numpy as np


# generate uniform data
def generate_uniform(n, B):
    file = open("uniform.txt", 'w')
    for _ in range(n):
        i = np.random.randint(0, B)
        file.write(str(i)+'\n')
    print("finish")
    file.close()


def generate_uniform_sample(n, B):
    file = open("uniform_sample.txt", 'w')
    for i in range(B):
        file.write(str(i)+'\n')
    print("finish")
    file.close()


def gen_gaussian_large(n, B):
    mean = B/2
    std = 1000000
    data = np.random.normal(mean, std, size=n)
    # t = data > B
    # print(np.where(data > B))
    file = open("gaussian.txt", "w")
    for i in data:
        file.write(str(int(i)) + '\n')
    print("finish")
    file.close()


class Zipfian:
    def __init__(self, alpha, B):
        self.alpha = alpha
        self.maxnoise = B

        self.accprob = []
        cumprob, prob = 0, 0

        for i in range(1, self.maxnoise):
            prob = 1.0 / pow(i, alpha)
            cumprob += prob
            self.accprob.append(cumprob)

        self.largeprob = self.accprob[-1]
        # print(self.accprob)

    def Generate(self):
        randness = random.uniform(0, self.largeprob)
        randnum = bisect.bisect_left(self.accprob, randness) + 1
        return randnum


def gen_zipf(n, B):
    alpha = 1.5
    zipf = Zipfian(alpha, B)
    zipflist = [[zipf.Generate()] for i in range(n)]
    # print (zipflist)
    file = open("zipf.txt", "w")
    for line in zipflist:
        file.write(str(line[0]) + '\n')
    print("finish")
    file.close()


def gen_gaussian(n, B):
    mean = B/3
    std = 20
    data = np.random.normal(mean, std, size=n)
    # t = data > B
    # print(np.where(data > B))
    file = open("gaussian.txt", "w")
    for i in data:
        file.write(str(int(i)) + '\n')
    print("finish")
    file.close()


def generate_2D(n):
    file = open("2d_uniform.txt", "w")
    # for i in range(32):
    #     for j in range(32):
    for i in range(n):
            i = np.random.randint(0, 32)
            j = np.random.randint(0, 32)
            file.write(str(int(i)) + " " + str(int(j)) + '\n')
    print("finish")
    file.close()


def generate_cube_sample(n):
    file = open("cube_sample.txt", "w")
    for x in range(8):
        for y in range(8):
            for m in range(4):
                for n in range(4):
                    file.write(str(int(x)) + " " + str(int(y)) + " " + str(int(m)) + " " + str(int(n)) + '\n')
    print("finish")
    file.close()


def generate_cube(n):
    file = open("cube.txt", "w")
    for _ in range(n):
        x = np.random.randint(0, 8)
        y = np.random.randint(0, 8)
        m = np.random.randint(0, 4)
        n = np.random.randint(0, 4)
        file.write(str(int(x)) + " " + str(int(y)) + " " + str(int(m)) + " " + str(int(n)) + '\n')
    print("finish")
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='optimal small domain range counting for shuffle model')
    parser.add_argument('--n', type=int, default=10000000, help='total number of user')
    parser.add_argument('--B', '--b', type=int, default=1024, help='domain range')
    parser.add_argument('--dataset', type=str, default='uniform',
                        help='input data set')
    parser.add_argument('--mode', help='mode for different queries')
    opt = parser.parse_args()
    n = opt.n
    b = opt.B
    t = opt.dataset
    mode = opt.mode
    if mode == "cube":
        generate_cube(n)
    elif mode == "multi":
        generate_2D(n)
    else:
        if t == "uniform":
            generate_uniform(n, b)
        elif t == "gaussian":
            gen_gaussian(n, b)
        else:
            gen_zipf(n, b)
