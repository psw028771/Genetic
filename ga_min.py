"""
以基因演算法 (Genetic Algorithm) 求
    f(x, y) = (x - y)^2 - x + 2y + sin(x + y) + 1
在 x ∈ [-1.5, 4], y ∈ [-3, 4] 的最小值。
x, y 精確至小數點後第 4 位；f 精確至小數點後第 5 位。

編碼：二進位染色體
    x 範圍長度 5.5，精度 1e-4 → 55000 個區間 → 需 16 bits (2^16 = 65536 ≥ 55001)
    y 範圍長度 7.0，精度 1e-4 → 70000 個區間 → 需 17 bits (2^17 = 131072 ≥ 70001)
選擇：錦標賽選擇；交配：單點交配；突變：位元翻轉；保留菁英。
"""
import math
import random

X_MIN, X_MAX = -1.5, 4.0
Y_MIN, Y_MAX = -3.0, 4.0
BITS_X, BITS_Y = 16, 17
CHROM_LEN = BITS_X + BITS_Y

POP_SIZE = 200
GENERATIONS = 500
P_CROSSOVER = 0.9
P_MUTATION = 1.0 / CHROM_LEN
ELITE = 2
TOURNAMENT_K = 3


def f(x, y):
    return (x - y) ** 2 - x + 2 * y + math.sin(x + y) + 1


def decode(chrom):
    """二進位字串 → (x, y)"""
    ix = int("".join(map(str, chrom[:BITS_X])), 2)
    iy = int("".join(map(str, chrom[BITS_X:])), 2)
    x = X_MIN + ix * (X_MAX - X_MIN) / (2 ** BITS_X - 1)
    y = Y_MIN + iy * (Y_MAX - Y_MIN) / (2 ** BITS_Y - 1)
    return x, y


def fitness(chrom):
    return f(*decode(chrom))  # 求最小值：值越小越好


def tournament(pop, fits):
    best = min(random.sample(range(len(pop)), TOURNAMENT_K), key=lambda i: fits[i])
    return pop[best]


def crossover(p1, p2):
    if random.random() < P_CROSSOVER:
        pt = random.randint(1, CHROM_LEN - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return p1[:], p2[:]


def mutate(chrom):
    return [1 - b if random.random() < P_MUTATION else b for b in chrom]


def ga(seed=None, verbose=True):
    random.seed(seed)
    pop = [[random.randint(0, 1) for _ in range(CHROM_LEN)] for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        fits = [fitness(c) for c in pop]
        order = sorted(range(POP_SIZE), key=lambda i: fits[i])
        new_pop = [pop[i][:] for i in order[:ELITE]]  # 菁英保留

        while len(new_pop) < POP_SIZE:
            c1, c2 = crossover(tournament(pop, fits), tournament(pop, fits))
            new_pop.append(mutate(c1))
            if len(new_pop) < POP_SIZE:
                new_pop.append(mutate(c2))
        pop = new_pop

        if verbose and (gen % 50 == 0 or gen == GENERATIONS - 1):
            bx, by = decode(pop[0])
            print(f"第 {gen:3d} 代: x = {bx:.4f}, y = {by:.4f}, f = {f(bx, by):.5f}")

    best = min(pop, key=fitness)
    return decode(best)


if __name__ == "__main__":
    x, y = ga(seed=1)
    x, y = round(x, 4), round(y, 4)
    print("\n=== 結果 ===")
    print(f"x = {x:.4f}")
    print(f"y = {y:.4f}")
    print(f"f(x, y) 最小值 = {f(x, y):.5f}")
