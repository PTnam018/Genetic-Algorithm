import random
import math
import matplotlib.pyplot as plt

# ============================================
#         GA CHUẨN (PERMUTATION GA)
#         GIẢI BÀI TOÁN TSP
# ============================================

class TSP_GA:
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

    # ------------------------------------------
    # TẠO MA TRẬN KHOẢNG CÁCH
    # ------------------------------------------
    def _build_distance_matrix(self):
        n = self.n_cities
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                x1, y1 = self.coords[i]
                x2, y2 = self.coords[j]
                dist[i][j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return dist

    # ------------------------------------------
    # QUẦN THỂ BAN ĐẦU (hoán vị)
    # ------------------------------------------
    def _random_individual(self):
        ind = list(range(self.n_cities))
        random.shuffle(ind)
        return ind

    # ------------------------------------------
    # TÍNH ĐỘ DÀI TOUR
    # ------------------------------------------
    def _tour_length(self, tour):
        total = 0.0
        n = len(tour)
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]  # quay lại điểm đầu
            total += self.dist[a][b]
        return total

    # FITNESS = 1 / length
    def _fitness(self, ind):
        return 1.0 / self._tour_length(ind)

    # ------------------------------------------
    # TOURNAMENT SELECTION
    # ------------------------------------------
    def _tournament_selection(self, population, fitnesses):
        best_idx = None
        for _ in range(self.tournament_size):
            idx = random.randrange(len(population))
            if (best_idx is None) or (fitnesses[idx] > fitnesses[best_idx]):
                best_idx = idx
        return population[best_idx][:]

    # ------------------------------------------
    # ORDER CROSSOVER (OX)
    # ------------------------------------------
    def _crossover_OX(self, p1, p2):
        n = len(p1)
        c1 = [-1] * n
        c2 = [-1] * n

        a, b = sorted(random.sample(range(n), 2))

        # Copy đoạn giữa
        c1[a:b] = p1[a:b]
        c2[a:b] = p2[a:b]

        # Hàm điền phần còn lại
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

    # ------------------------------------------
    # SWAP MUTATION
    # ------------------------------------------
    def _mutate_swap(self, ind):
        if random.random() < self.pm:
            i, j = random.sample(range(len(ind)), 2)
            ind[i], ind[j] = ind[j], ind[i]
        return ind

    # ------------------------------------------
    # CHẠY THUẬT TOÁN GA
    # ------------------------------------------
    def run(self, verbose=True):
        population = [self._random_individual() for _ in range(self.pop_size)]

        best_ind = None
        best_length = float('inf')
        convergence = []

        for gen in range(self.generations):
            fitnesses = [self._fitness(ind) for ind in population]

            # Cập nhật best
            for ind in population:
                length = self._tour_length(ind)
                if length < best_length:
                    best_length = length
                    best_ind = ind[:]

            convergence.append(best_length)

            if verbose and gen % 50 == 0:
                print(f"Generation {gen}: best length = {best_length:.4f}")

            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)

                # lai
                if random.random() < self.pc:
                    c1, c2 = self._crossover_OX(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                # đột biến
                new_pop.append(self._mutate_swap(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate_swap(c2))

            population = new_pop

        # -------- KẾT QUẢ CUỐI CÙNG --------
        print("===== KẾT QUẢ =====")
        print("Best tour:", best_ind)
        print("Best length:", best_length)

        # Plot hội tụ như RCGA
        plt.figure(figsize=(8, 5))
        plt.plot(convergence, label="Best length")
        plt.xlabel("Generation")
        plt.ylabel("Tour length")
        plt.title("Convergence of TSP GA (Permutation Encoding)")
        plt.grid(True)
        plt.legend()
        plt.show()

        return best_ind, best_length


if __name__ == "__main__":
    random.seed(42)
    n_cities = 10
    coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_cities)]

    ga = TSP_GA(coords,
                pop_size=60,
                generations=500,
                pc=0.9,
                pm=0.1,
                tournament_size=3)

    best_tour, best_length = ga.run(verbose=True)