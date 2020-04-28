import numpy as np
import random


def make(n, k):
    cl = [random.randint(n*4/k/5, n*5/k/4) for _ in range(k)]
    s = np.sum(cl)
    if s > n:
        if cl[k-1] - s + n > 0:
            cl[k-1] = cl[k-1] - s + n
    else:
        cl[k-1] += n - s

    file_path = 'graph' + str(n) + '.vis'
    with open(file_path, 'w') as f:
        f.write("%s\n" % k)
        for i in range(k):
            f.write("%s " % cl[i])

    dd = [0] * k

    for i in range(k):
        dd[i] = cl[i]*int(np.sqrt(cl[i]))*5/6

    add = [max(_/4, 2) for _ in cl]

    m = np.sum(dd) + np.sum(add)

    ok = np.zeros(shape=(n, n), dtype=int)
    cnt = 0
    file_path = 'graph' + str(n) + '.txt'
    with open(file_path, 'w') as f:
        f.write("%s %s %s\n" % (n, m, k))
        for i in range(k):
            for j in range(dd[i]):
                while True:
                    u = random.randint(cnt, cnt + cl[i] - 1)
                    v = random.randint(cnt, cnt + cl[i] - 1)
                    if (u != v) and (ok[u][v] == 0):
                        f.write("%s %s\n" % (u, v))
                        ok[u][v] = 1
                        ok[v][u] = 1
                        break
            for j in range(add[i]):
                while True:
                    u = random.randint(cnt, cnt + cl[i] - 1)
                    if i == k-1:
                        v = random.randint(0, cl[0]-1)
                    else:
                        v = random.randint(cnt+cl[i], cnt+cl[i]+cl[i+1] - 1)
                    if (u != v) and (ok[u][v] == 0):
                        # if i == k-1 and j == add[i]-1:
                        #     f.write("%s %s" % (u, v))
                        # else:
                        f.write("%s %s\n" % (u, v))
                        ok[u][v] = 1
                        break
            cnt += cl[i]

make(600, 20)


