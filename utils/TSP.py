import numpy as np
import time


class TSP:
    def __init__(self, n, dist_mat):
        """
        dist_mat: (n+1) * (n+1)  Need contains the depot in index 0 and index n.
        """
        self.n = n - 1  # -end
        self.dp = np.ones((2 ** (n + 1), n)) * -1
        self.path = np.ones((2 ** (n + 1), n))
        self.dist_mat = dist_mat  # np.zeros((n, n))

    def tsp(self, s, init):
        if self.dp[s][init] != -1:
            return self.dp[s][init]
        if s == (1 << self.n):
            return self.dist_mat[self.n][init]  # to the end
            # return 0
            # return self.dist_mat[0][init]
        sum_path = float('inf')
        for i in range(self.n):
            if s & (1 << i):
                m = self.tsp(s & (~(1 << i)), i) + self.dist_mat[i][init]
                if m < sum_path:
                    sum_path = m
                    self.path[s][init] = i
        self.dp[s][init] = sum_path
        return self.dp[s][init]

    def solve(self):
        if self.n == 1:
            return self.dist_mat[0][1], [0, 1]

        init_point = 0
        s = 0
        for i in range(1, self.n + 1):
            s = s | (1 << i)

        # start = time.time()
        distance = self.tsp(s, init_point)
        # end = time.time()

        # print(distance)
        # print("TSP cost time: %ss" % (end - start))

        visit_seq = []
        s = 0
        for i in range(1, self.n + 1):
            s = s | (1 << i)
        init = 0
        num = 0
        visit_seq.append(0)
        while True:
            # print(visit_seq)
            visit_seq.append(int(self.path[s][init]))
            init = int(self.path[s][init])
            s = s & (~(1 << init))
            num += 1
            if s == (1 << self.n):
                break
        visit_seq.append(self.n)
        # print(visit_seq)
        return distance, visit_seq  # include START and END


if __name__ == "__main__":
    pass
