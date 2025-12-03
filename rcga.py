import random
import math
import itertools
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
            low, high = cmin - self.alpha * I, cmax + self.alpha * I
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

    # ================================
    # TÍNH GIÁ TRỊ ĐÚNG (BRUTE FORCE)
    # ================================
    def compute_optimal_solution(self):
        print("Đang brute-force nghiệm đúng... có thể mất vài giây.")
        cities = list(range(self.n_cities))
        best_length = float('inf')
        best_tour = None

        for perm in itertools.permutations(cities):
            length = self._tour_length(perm)
            if length < best_length:
                best_length = length
                best_tour = perm

        print(">>> Giá trị đúng (Optimal):", best_length)
        return best_length, best_tour

    # ================================
    #          CHẠY GIẢI THUẬT
    # ================================
    def run(self, verbose=True):

        # Tính nghiệm đúng trước
        optimal_value, optimal_tour = self.compute_optimal_solution()

        population = [self._random_individual() for _ in range(self.pop_size)]
        best_length = float('inf')
        convergence = []

        for gen in range(self.generations):

            fitnesses = [self._fitness(ind) for ind in population]

            # Lấy best
            for ind in population:
                tour = self._decode(ind)
                length = self._tour_length(tour)
                if length < best_length:
                    best_length = length

            convergence.append(best_length)

            # Tạo quần thể mới
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)

                if random.random() < self.pc:
                    c1, c2 = self._crossover_blx_alpha(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                new_pop.append(self._mutate_gaussian(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate_gaussian(c2))

            population = new_pop

        # =============================
        # HIỆN SƠ ĐỒ HỘI TỤ
        # =============================
        plt.figure(figsize=(9, 5))
        plt.plot(convergence, label="RCGA Best Length")
        plt.axhline(optimal_value, color="red", linestyle="--", linewidth=2, label="Optimal Value")
        plt.xlabel("Generation")
        plt.ylabel("Tour Length")
        plt.title("RCGA Convergence vs Optimal")
        plt.grid(True)
        plt.legend()
        plt.show()

        return optimal_value, best_length, convergence


# =========================
#  CHẠY THỬ
# =========================
if __name__ == "__main__":
    random.seed(42)
    n_cities = 9     # brute-force cần nhỏ (<10)
    coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_cities)]

    ga = TSP_RCGA(coords, pop_size=60, generations=300)
    optimal, best_ga, conv = ga.run()
