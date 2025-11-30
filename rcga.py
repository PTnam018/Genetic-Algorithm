import random
import math
import matplotlib.pyplot as plt

# =========================
#  RCGA cho bài toán TSP
# =========================

class TSP_RCGA:
    def __init__(self, coords, pop_size=50, generations=500,
                 pc=0.9, pm=0.1, alpha=0.3, sigma=0.1, tournament_size=3):
        self.coords = coords
        self.n_cities = len(coords)
        self.pop_size = pop_size
        self.generations = generations
        self.pc = pc
        self.pm = pm
        self.alpha = alpha
        self.sigma = sigma
        self.tournament_size = tournament_size
        self.dist = self._build_distance_matrix()

    def _build_distance_matrix(self):
        n = self.n_cities
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                x1, y1 = self.coords[i]
                x2, y2 = self.coords[j]
                dist[i][j] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return dist

    def _random_individual(self):
        return [random.random() for _ in range(self.n_cities)]

    def _decode(self, individual):
        indexed = list(enumerate(individual))
        sorted_by_key = sorted(indexed, key=lambda x: x[1])
        tour = [idx for idx, key in sorted_by_key]
        return tour

    def _tour_length(self, tour):
        total = 0.0
        n = len(tour)
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]
            total += self.dist[a][b]
        return total

    def _fitness(self, individual):
        length = self._tour_length(self._decode(individual))
        return 1.0 / length if length != 0 else float('inf')

    def _tournament_selection(self, population, fitnesses):
        best_idx = None
        for _ in range(self.tournament_size):
            idx = random.randrange(len(population))
            if (best_idx is None) or (fitnesses[idx] > fitnesses[best_idx]):
                best_idx = idx
        return population[best_idx][:]

    def _crossover_blx_alpha(self, p1, p2):
        c1, c2 = [], []
        for x1, x2 in zip(p1, p2):
            cmin, cmax = min(x1, x2), max(x1, x2)
            I = cmax - cmin
            low, high = cmin - self.alpha*I, cmax + self.alpha*I
            v1, v2 = random.uniform(low, high), random.uniform(low, high)
            c1.append(min(1.0, max(0.0, v1)))
            c2.append(min(1.0, max(0.0, v2)))
        return c1, c2

    def _mutate_gaussian(self, individual):
        for i in range(len(individual)):
            if random.random() < self.pm:
                individual[i] += random.gauss(0, self.sigma)
                individual[i] = min(1.0, max(0.0, individual[i]))
        return individual

    def run(self, verbose=True):
        population = [self._random_individual() for _ in range(self.pop_size)]

        best_individual = None
        best_length = float('inf')
        best_tour = None
        convergence = []  # lưu độ dài tốt nhất theo thế hệ

        for gen in range(self.generations):
            fitnesses = [self._fitness(ind) for ind in population]

            # Cập nhật best
            for ind, fit in zip(population, fitnesses):
                tour = self._decode(ind)
                length = self._tour_length(tour)
                if length < best_length:
                    best_length = length
                    best_individual = ind[:]
                    best_tour = tour[:]

            convergence.append(best_length)

            if verbose and gen % 50 == 0:
                print(f"Generation {gen}: best length = {best_length:.4f}")

            # Tạo quần thể mới
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)

                if random.random() < self.pc:
                    c1, c2 = self._crossover_blx_alpha(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                c1 = self._mutate_gaussian(c1)
                c2 = self._mutate_gaussian(c2)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            population = new_pop

        # Hiển thị kết quả
        if verbose:
            print("===== KẾT QUẢ =====")
            print("Best tour:", best_tour)
            print("Best length:", best_length)

            # Vẽ sơ đồ hội tụ
            plt.figure(figsize=(8,5))
            plt.plot(convergence, label="Best length")
            plt.xlabel("Generation")
            plt.ylabel("Tour length")
            plt.title("Convergence of TSP RCGA")
            plt.grid(True)
            plt.legend()
            plt.show()

        return best_tour, best_length, best_individual


# =========================
#  Ví dụ chạy thử
# =========================
if __name__ == "__main__":
    random.seed(42)
    n_cities = 10
    coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_cities)]

    ga = TSP_RCGA(coords,
                  pop_size=60,
                  generations=500,
                  pc=0.9,
                  pm=0.1,
                  alpha=0.3,
                  sigma=0.1,
                  tournament_size=3)

    best_tour, best_length, best_individual = ga.run(verbose=True)
