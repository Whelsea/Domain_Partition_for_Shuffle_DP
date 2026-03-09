import math
import numpy as np
import random


class GKMPS:
    def __init__(self, n, domain, epsilon, delta, gamma):
        self.n = n  # the number of users
        self.domain = domain  # Domain size
        self.name = "GKMPS"  # name for hsdp
        scale = 2  # change the scale factor for different bucket sizes
        if domain > math.ceil(epsilon * math.sqrt(n) * scale):
            self.U = math.ceil(epsilon * math.sqrt(n) * scale)  # try a small domain
            self.B = math.ceil(domain / self.U)  # bucket size
        else:
            self.U = domain
            self.B = 1
        self.epsilon = epsilon
        self.delta = delta
        self.epsilonstar = (1 - gamma) * epsilon
        self.epsilon1 = min(1.0, gamma * epsilon) * 0.5
        self.epsilon2 = min(1.0, gamma * epsilon) * 0.5
        self.delta1 = delta / 2
        self.delta2 = delta / 2

        self.t = np.zeros(2 * self.U + 1)  # self.t is an array of size [-U, U]
        Gamma = self.U * (math.ceil(np.log(self.U)) + 1)
        for i in range(-self.U, self.U + 1, 1):
            if i != 0:
                self.t[i] = math.ceil(Gamma / abs(i))

    def RandomizedRounding(self, x):
        # Make sure value is below U
        x = max(0, x)
        x = min(x, self.domain)

        x = x / self.B
        prob = x - math.floor(x)
        if random.random() <= prob:
            x = math.ceil(x)
        else:
            x = math.floor(x)
        return x

    ###############################################################################################################
    #
    #   LocalRandomizer 
    #   Input: value (the user hold) 
    #   Output: a list of messages the user sent to secure shuffler
    #
    ###############################################################################################################
    def LocalRandomizer(self, value):
        # real mesasge
        messages = []
        if value != 0:
            messages.append(self.RandomizedRounding(value))

        # central noise
        num_plus_1 = np.random.negative_binomial(1.0 / self.n, 1 - math.exp(-self.epsilonstar / self.U))
        messages += [1] * num_plus_1
        num_minus_1 = np.random.negative_binomial(1.0 / self.n, 1 - math.exp(-self.epsilonstar / self.U))
        messages += [-1] * num_minus_1

        # sum zero noises
        z = np.zeros(2 * self.U + 1)

        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            if i == 1:
                z[i] += np.random.negative_binomial(3 * (1 + np.log(1 / self.delta1)) / self.n,
                                                    1 - math.exp(-0.2 * self.epsilon1 / self.U))
            z[i] += np.random.negative_binomial(3 * (1 + np.log((2 * self.U - 1) / self.delta2)) / self.n,
                                                1 - math.exp(-0.1 * self.epsilon2 / self.t[i]))

        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            elif i == 1:
                messages += [-1, 1] * int(z[i])
            else:
                messages += [i, (- i // 2), (- i - (-i // 2))] * int(z[i])

        return messages

    ###############################################################################################################
    #
    #   Analyzer 
    #   Input: a list of messages from all users (ignore shuffling)
    #   Output: sum of the messages
    #
    ###############################################################################################################
    def Analyzer(self, messages, values=''):
        if values == '':
            return sum(messages) * self.B

        # Debug
        rrvalues = []
        for value in values:
            rrvalues.append(self.RandomizedRounding(value))

        accurate_sum = sum(values)
        rr_sum = sum(rrvalues) * self.B
        dp_sum = sum(messages) * self.B
        print("accurate sum = ", accurate_sum)
        print("random rounding sum = ", rr_sum)
        print("dp sum = ", dp_sum)
        print("    |DP - ACC| = ", abs(dp_sum - accurate_sum))
        print("    |DP - RR| = ", abs(dp_sum - rr_sum))
        print("    |RR - ACC| = ", abs(rr_sum - accurate_sum))
        print("#messages/user = ", len(messages) / len(values))

        return dp_sum

    ###############################################################################################################
    #
    #   EstimateMessageNumber 
    #   Input: value (the user hold) 
    #   Output: the number of messages the user sent (expected)
    #
    ###############################################################################################################
    def EstimateMessageNumber(self, value=''):
        message = 0

        # need to send real mesasge
        if value != 0 and value != '':
            message += 0.5

        # central noise
        message += 2 * (1.0 / self.n * self.U / self.epsilonstar)

        # sum zero noises
        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            if i == 1:
                message += 2 * (3 * (1 + np.log(1 / self.delta1)) / self.n * self.U / (0.2 * self.epsilon1))
            message += 3 * (
                    3 * (1 + np.log((2 * self.U - 1) / self.delta2)) / self.n * self.t[i] / (0.1 * self.epsilon2))
        return message

    ###############################################################################################################
    #
    #   Simulator 
    #   Input: a list of input values
    #   Output: the total number of received messages, the DP noised sum (in practice)
    #
    ###############################################################################################################
    def Simulator(self, values):
        # Real values
        nmessages, dpsum = 0, 0
        for value in values:
            rvalue = self.RandomizedRounding(value)
            if (rvalue != 0):
                nmessages += 1
            dpsum += rvalue

        # Central-DP values
        print(1 - math.exp(-self.epsilonstar / self.U))
        print(self.epsilonstar)
        print(self.U)
        num_plus_1 = np.random.negative_binomial(1.0, 1 - math.exp(-self.epsilonstar / self.U))
        num_minus_1 = np.random.negative_binomial(1.0, 1 - math.exp(-self.epsilonstar / self.U))
        nmessages += num_plus_1 + num_minus_1
        dpsum = dpsum + num_plus_1 - num_minus_1

        # Zero-sum DP values
        z = np.zeros(2 * self.U + 1)
        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            if i == 1:
                z[i] += np.random.negative_binomial(3 * (1 + np.log(1 / self.delta1)),
                                                    1 - math.exp(-0.2 * self.epsilon1 / self.U))
            z[i] += np.random.negative_binomial(3 * (1 + np.log((2 * self.U - 1) / self.delta2)),
                                                1 - math.exp(-0.1 * self.epsilon2 / self.t[i]))

        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            elif i == 1:
                nmessages += 2 * int(z[i])
            else:
                nmessages += 3 * int(z[i])

        return nmessages, dpsum * self.B

    ###############################################################################################################
    #
    #   Simulator for HSDP
    #   Difference: doubles noise
    #
    ###############################################################################################################
    def Simulator_for_HSDP(self, values, honest_user_proportion):
        # Real values
        nmessages, dpsum = 0, 0
        for value in values:
            rvalue = self.RandomizedRounding(value)
            if (rvalue != 0):
                nmessages += 1
            dpsum += rvalue

        # Central-DP values
        num_plus_1 = np.random.negative_binomial(2.0 * honest_user_proportion, 1 - math.exp(-self.epsilonstar / self.U))
        num_minus_1 = np.random.negative_binomial(2.0 * honest_user_proportion,
                                                  1 - math.exp(-self.epsilonstar / self.U))

        nmessages += num_plus_1 + num_minus_1
        dpsum = dpsum + num_plus_1 - num_minus_1

        # Zero-sum DP values
        z = np.zeros(2 * self.U + 1)
        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            if i == 1:
                z[i] += np.random.negative_binomial(6 * honest_user_proportion * (1 + np.log(1 / self.delta1)),
                                                    1 - math.exp(-0.2 * self.epsilon1 / self.U))
            z[i] += np.random.negative_binomial(
                6 * honest_user_proportion * (1 + np.log((2 * self.U - 1) / self.delta2)),
                1 - math.exp(-0.1 * self.epsilon2 / self.t[i]))

        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            elif i == 1:
                nmessages += 2 * int(z[i])
            else:
                nmessages += 3 * int(z[i])

        return nmessages, dpsum * self.B

    ###############################################################################################################
    #
    #   Simulator for GKMPS when k>0
    #   Difference: doubles noise
    #
    ###############################################################################################################
    def Simulator_for_GKMPS_k(self, values, honest_user_proportion):
        # Real values
        nmessages, dpsum = 0, 0
        for value in values:
            rvalue = self.RandomizedRounding(value)
            if (rvalue != 0):
                nmessages += 1
            dpsum += rvalue
        # print(f"U: {self.U}")
        # print(f"Bits per message: {1 + math.ceil(math.log2(self.U + 1))}")
        # Central-DP values
        num_plus_1 = np.random.negative_binomial(honest_user_proportion, 1 - math.exp(-self.epsilonstar / self.U))
        num_minus_1 = np.random.negative_binomial(honest_user_proportion,
                                                  1 - math.exp(-self.epsilonstar / self.U))
        nmessages += num_plus_1 + num_minus_1
        # print("central", (num_plus_1 + num_minus_1)/len(values))
        dpsum = dpsum + num_plus_1 - num_minus_1
        # Zero-sum DP values
        z = np.zeros(2 * self.U + 1)
        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            if i == 1:
                z[i] += np.random.negative_binomial(3 * honest_user_proportion * (1 + np.log(1 / self.delta1)),
                                                    1 - math.exp(-0.2 * self.epsilon1 / self.U))
                # print(f"delta:{self.delta1},count1:{z[i]/len(values)}")

            count = np.random.negative_binomial(
                3 * honest_user_proportion * (1 + np.log((2 * self.U - 1) / self.delta2)),
                1 - math.exp(-0.1 * self.epsilon2 / self.t[i]))
            z[i] += count
            # print(f"count2:{count / len(values)}")
        for i in range(-self.U, self.U + 1, 1):
            if i == 0 or i == -1:
                continue
            elif i == 1:
                nmessages += 2 * int(z[i])

            else:
                nmessages += 3 * int(z[i])

        return nmessages, dpsum * self.B


def main():
    # ==== 参数设定 ====
    n = 2**12
    eps = 1.0
    delta = 1 / (n * n)
    domain = 2
    gamma = 0.3

    # ==== 三层隐私预算划分 ====
    d = domain - 1

    eps_user = eps / 3
    eps_block = (eps / 3) * ((math.sqrt(n) - 1) / math.sqrt(n))
    eps_output = (eps / 3) * ((n - 1) / n)

    delta_user = delta / 3
    delta_block = (delta / 3) * ((math.sqrt(n) - 1) / math.sqrt(n))
    delta_output = (delta / 3) * ((n - 1) / n)

    # ==== 初始化三层 GKMPS ====
    gkmps_user = GKMPS(n=1, domain=d, epsilon=eps_user, delta=delta_user, gamma=gamma)
    gkmps_block = GKMPS(n=int(math.sqrt(n)), domain=d, epsilon=eps_block, delta=delta_block, gamma=gamma)
    gkmps_output = GKMPS(n=n, domain=d, epsilon=eps_output, delta=delta_output, gamma=gamma)

    print("===== 三层 GKMPS 初始化信息 =====")
    for name, g in zip(["User", "Block", "Output"], [gkmps_user, gkmps_block, gkmps_output]):
        print(f"[{name}] n={g.n}, U={g.U}, ε={g.epsilon:.4f}, ε*={g.epsilonstar:.4f}, δ={g.delta:.2e}")

    print("=" * 50)
    # ==== 理论期望消息数 ====
    est_user = gkmps_user.EstimateMessageNumber(1)
    est_block = gkmps_block.EstimateMessageNumber(1)
    est_output = gkmps_output.EstimateMessageNumber(1)

    total_msgs = n * est_user + math.sqrt(n) * est_block + est_output
    avg_per_user = total_msgs / n

    print(f"[User layer]    每用户 ≈ {est_user:.2f}")
    print(f"[Block layer]   每block ≈ {est_block:.2f}")
    print(f"[Output layer]  全局 ≈ {est_output:.2f}")
    print("-" * 50)
    print(f"[Total expected messages] ≈ {total_msgs:.2f}")
    print(f"[Average per user] ≈ {avg_per_user:.2f}")
    print("=" * 50)

    # ==== 模拟检查 ====
    values = [1] * n
    m_user, _ = gkmps_user.Simulator(values[:1])
    m_block, _ = gkmps_block.Simulator(values[:int(math.sqrt(n))])
    m_output, _ = gkmps_output.Simulator(values)
    total_sim =  m_user +m_block + m_output
    print(f"[Simulator] 实际观测消息总数 ≈ {total_sim:.2f}")

# def main():
#     # ==== 参数设定 ====
#     n = 1
#     eps = 1
#     actn=2**24
#     delta = 1 / (actn * actn)
#     domain = 1
#     gamma = 0.3
#
#     # ==== 初始化 GKMPS ====
#     gkmps = GKMPS(n=n, domain=domain, epsilon=eps, delta=delta, gamma=gamma)
#
#     # ==== 打印基本参数 ====
#     print("===== GKMPS 初始化信息 =====")
#     print(f"n = {gkmps.n}")
#     print(f"domain = {gkmps.domain}")
#     print(f"U = {gkmps.U}")
#     print(f"B = {gkmps.B}")
#     print(f"ε = {gkmps.epsilon}")
#     print(f"ε* = {gkmps.epsilonstar}")
#     print(f"ε1 = {gkmps.epsilon1}, ε2 = {gkmps.epsilon2}")
#     print(f"δ1 = {gkmps.delta1}, δ2 = {gkmps.delta2}")
#     print("=" * 40)
#
#     # ==== 理论估计 ====
#     est = gkmps.EstimateMessageNumber(1)
#     print(f"[Estimate] 期望消息数 ≈ {est:.2f}")
#
#     # ==== 实际模拟 ====
#     values = [1]  # 单用户
#     nmessages, dpsum = gkmps.Simulator(values)
#     print(f"[Simulator] 实际发送消息数 = {nmessages}")
#     print(f"[Simulator] 最终 DP 求和结果 = {dpsum}")
#
#     # ==== 再试几次取平均 ====
#     trials = 10
#     message_counts = []
#     for _ in range(trials):
#         m, _ = gkmps.Simulator(values)
#         message_counts.append(m)
#     print(f"[Average over {trials} runs] 平均消息数 ≈ {np.mean(message_counts):.2f}")

if __name__ == "__main__":
    main()