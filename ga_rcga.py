import random
import math
import itertools
import matplotlib.pyplot as plt

# ============================================================
# TSP GA PERMUTATION (hoán vị)
# ============================================================
class TSP_GA_Permutation:
    def __init__(self, coords, pop_size=60, generations=500,
                 pc=0.9, pm=0.1, tournament_size=3):
        self.coords = coords
        self.n_cities = len(coords)
        self.pop_size = pop_size
        self.generations = generations
        self.pc = pc
        self.pm = pm
        self.tournament_size = tournament_size
        self.dist = self._build_distance_matrix()

    def _build_distance_matrix(self):
        n = self.n_cities
        dist = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                x1, y1 = self.coords[i]
                x2, y2 = self.coords[j]
                dist[i][j] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return dist

    def _random_individual(self):
        ind = list(range(self.n_cities))
        random.shuffle(ind)
        return ind

    def _tour_length(self, tour):
        total = 0.0
        n = len(tour)
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]
            total += self.dist[a][b]
        return total

    def _fitness(self, ind):
        return 1.0 / self._tour_length(ind)

    def _tournament_selection(self, population, fitnesses):
        best_idx = None
        for _ in range(self.tournament_size):
            idx = random.randrange(len(population))
            if (best_idx is None) or (fitnesses[idx] > fitnesses[best_idx]):
                best_idx = idx
        return population[best_idx][:]

    def _crossover_OX(self, p1, p2):
        n = len(p1)
        c1, c2 = [-1]*n, [-1]*n
        a, b = sorted(random.sample(range(n), 2))
        c1[a:b] = p1[a:b]
        c2[a:b] = p2[a:b]

        def fill(child, parent):
            pos = b
            for city in parent:
                if city not in child:
                    if pos >= n:
                        pos = 0
                    child[pos] = city
                    pos += 1
        fill(c1, p2)
        fill(c2, p1)
        return c1, c2

    def _mutate_swap(self, ind):
        if random.random() < self.pm:
            i, j = random.sample(range(len(ind)), 2)
            ind[i], ind[j] = ind[j], ind[i]
        return ind

    def compute_optimal(self):
        # Tính giá trị đúng (brute-force)
        cities = list(range(self.n_cities))
        best_len = float('inf')
        best_tour = None
        for perm in itertools.permutations(cities):
            L = self._tour_length(perm)
            if L < best_len:
                best_len = L
                best_tour = perm
        return best_len, best_tour

    def run(self, verbose=True):
        population = [self._random_individual() for _ in range(self.pop_size)]
        best_ind = None
        best_length = float('inf')
        convergence = []

        for gen in range(self.generations):
            fitnesses = [self._fitness(ind) for ind in population]

            for ind in population:
                L = self._tour_length(ind)
                if L < best_length:
                    best_length = L
                    best_ind = ind[:]

            convergence.append(best_length)  # Lưu best length của thế hệ

            # Tạo quần thể mới
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)
                if random.random() < self.pc:
                    c1, c2 = self._crossover_OX(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                new_pop.append(self._mutate_swap(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate_swap(c2))
            population = new_pop

        return best_ind, best_length, convergence


# ============================================================
# TSP RCGA (Real-Coded GA)
# ============================================================
class TSP_RCGA:
    def __init__(self, coords, pop_size=60, generations=500,
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
        dist = [[0.0]*n for _ in range(n)]
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
        return [idx for idx, key in sorted_by_key]

    def _tour_length(self, tour):
        total = 0.0
        n = len(tour)
        for i in range(n):
            a = tour[i]
            b = tour[(i+1)%n]
            total += self.dist[a][b]
        return total

    def _fitness(self, individual):
        L = self._tour_length(self._decode(individual))
        return 1.0 / L

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
            c1.append(min(1.0,max(0.0,v1)))
            c2.append(min(1.0,max(0.0,v2)))
        return c1, c2

    def _mutate_gaussian(self, individual):
        for i in range(len(individual)):
            if random.random() < self.pm:
                individual[i] += random.gauss(0,self.sigma)
                individual[i] = min(1.0,max(0.0,individual[i]))
        return individual

    def run(self):
        population = [self._random_individual() for _ in range(self.pop_size)]
        best_length = float('inf')
        convergence = []

        for gen in range(self.generations):
            fitnesses = [self._fitness(ind) for ind in population]
            for ind in population:
                L = self._tour_length(self._decode(ind))
                if L < best_length:
                    best_length = L
            convergence.append(best_length)

            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)
                if random.random() < self.pc:
                    c1, c2 = self._crossover_blx_alpha(p1,p2)
                else:
                    c1,c2 = p1[:],p2[:]
                new_pop.append(self._mutate_gaussian(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate_gaussian(c2))
            population = new_pop

        return best_length, convergence


# ============================================================
# MAIN: So sánh GA Permutation vs RCGA
# ============================================================
if __name__ == "__main__":
    random.seed(42)
    n_cities = 10
    coords = [(random.uniform(0,100), random.uniform(0,100)) for _ in range(n_cities)]

    # --- Tính nghiệm đúng (brute-force) ---
    ga_perm = TSP_GA_Permutation(coords, pop_size=60, generations=500)
    best_len_true, best_tour_true = ga_perm.compute_optimal()
    print("Brute-force best length (true optimal):", best_len_true)

    # --- GA Permutation ---
    ga_perm = TSP_GA_Permutation(coords, pop_size=60, generations=500)
    best_tour_ga, best_len_ga, conv_ga = ga_perm.run()[:3]

    # --- RCGA ---
    rcga = TSP_RCGA(coords, pop_size=60, generations=500)
    best_len_rcga, conv_rcga = rcga.run()

    # --- Vẽ biểu đồ hội tụ ---
    plt.figure(figsize=(9,5))
    plt.plot(conv_ga, label="GA Permutation Best Length", linewidth=2)
    plt.plot(conv_rcga, label="RCGA Best Length", linewidth=2)
    plt.axhline(y=best_len_true, color='red', linestyle='--', linewidth=2, label="True Optimal Length")

    plt.xlabel("Generation")
    plt.ylabel("Best tour length")
    plt.title("Convergence: GA Permutation vs RCGA")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Kết quả cuối ---
    print("\n===== KẾT QUẢ CUỐI =====")
    print("True optimal length (brute-force):", best_len_true)
    print("GA Permutation best length:", best_len_ga)
    print("RCGA best length:", best_len_rcga)
