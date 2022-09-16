import math
import matplotlib.pyplot as plt


class lcg:  # 线性同余法随机数生成器：结果服从$U(0,1)$。

    def __init__(self, seed, size, k=32, a=5, c=0):
        self.state = seed  # 初始值$x_0$
        self.index = 0  # 记录生成数下标
        self.n = size  # 生成数量
        self.M = pow(2, k)  # 模数$M$
        self.a = a  # 乘数
        self.c = c  # 加数，乘同余法中为零

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.n:
            # $x_{i+1} = (Ax_{i}+c)\ (mod\ M)$
            self.state = (self.state * self.a + self.c) % self.M
            self.index += 1
            return self.state / self.get_range()  # 归一化
        else:
            raise StopIteration

    def get_range(self):
        return self.M  # 获得原始随机数的范围


# 逆变法随机数生成器：结果服从参数为$\lambda$的负指数分布
def expo_distribution_generator(l, sd=1, size=100):
    prng = lcg(sd, size)
    for u in prng:
        yield -1/l * math.log(u)  # 逆变法，$x=-\frac{1}{\lambda}\ln u$


if __name__ == '__main__':
    n = 2000
    seed = 16579875
    lmd = 5

    prn = [x for x in expo_distribution_generator(lmd, seed, n)]

    print("Average: ", sum(prn) / n)  # 计算随机数的均值, 结果应当接近$\frac{1}{\lambda}$

    # 作图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=[0, n], ylim=[0, 1.75], title='Exponential distribution',
           ylabel='Y-Axis', xlabel='X-Axis')
    x = list(range(n))
    plt.scatter(x, prn)
    plt.savefig('../fig/Exponential distribution scatter plot.pdf')
    plt.show()
