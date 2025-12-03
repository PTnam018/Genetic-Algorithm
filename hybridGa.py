import random
import math
import itertools
import matplotlib.pyplot as plt

# ============================
#  Hybrid GA cho bài toán TSP
# ============================

# ----- Hàm cơ bản -----
def distance_matrix(coords):
    n = len(coords)
    dist = [[0.0]*n for _ in range(n)]
    for i in range(n):
        x1, y1 = coords[i]
        for j in range(n):
            x2, y2 = coords[j]
            dist[i][j] = math.hypot(x1 - x2, y1 - y2)
    return dist

def tour_length(tour, dist):
    return sum(dist[tour[i]][tour[(i+1)%len(tour)]] for i in range(len(tour)))

def fitness(tour, dist):
    L = tour_length(tour, dist)
    return float("inf") if L == 0 else 1.0 / L

# ----- GA operators -----
def random_tour(n_cities):
    tour = list(range(n_cities))
    random.shuffle(tour)
    return tour

def tournament_selection(population, fitnesses, k=3):
    best_idx = None
    for _ in range(k):
        idx = random.randrange(len(population))
        if best_idx is None or fitnesses[idx] > fitnesses[best_idx]:
            best_idx = idx
    return population[best_idx][:]

def order_crossover(p1, p2):
    size = len(p1)
    c1, c2 = [None]*size, [None]*size
    i, j = sorted(random.sample(range(size), 2))

    c1[i:j+1] = p1[i:j+1]
    c2[i:j+1] = p2[i:j+1]

    def fill(child, parent):
        pos = (j+1) % size
        for gene in parent:
            if gene not in child:
                child[pos] = gene
                pos = (pos+1) % size

    fill(c1, p2)
    fill(c2, p1)
    return c1, c2

def swap_mutation(tour, pm):
    if random.random() < pm:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

# ----- Local Search: 2-opt -----
def two_opt(tour, dist):
    best = tour[:]
    best_len = tour_length(tour, dist)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                new_tour = best[:]
                new_tour[i:j+1] = reversed(new_tour[i:j+1])
                new_len = tour_length(new_tour, dist)
                if new_len < best_len:
                    best, best_len = new_tour, new_len
                    improved = True
    return best

# ============================================
#     TÍNH GIÁ TRỊ ĐÚNG (BRUTE FORCE)
# ============================================
def compute_optimal_solution(n_cities, dist):
    print("\n>>> Đang brute-force... ")
    best_len = float("inf")
    best_tour = None

    for perm in itertools.permutations(range(n_cities)):
        L = tour_length(perm, dist)
        if L < best_len:
            best_len = L
            best_tour = perm

    print(">>> Optimal length =", best_len)
    print(">>> Optimal tour   =", best_tour)
    return best_len, best_tour

# ----- Hybrid GA chính -----
def hybrid_ga_tsp(coords, pop_size=80, generations=400, pc=0.9, pm=0.2,
                  tournament_k=3, local_search_rate=1.0, verbose=True,
                  max_plot_gen=100, convergence_interval=5):

    n_cities = len(coords)
    dist = distance_matrix(coords)

    # ======= TÍNH NGHIỆM ĐÚNG TRƯỚC =======
    optimal_len, optimal_tour = compute_optimal_solution(n_cities, dist)

    # ======= KHỞI TẠO QUẦN THỂ =======
    population = [random_tour(n_cities) for _ in range(pop_size)]
    best_tour = None
    best_len = float("inf")
    convergence = []
    gen_list = []

    for gen in range(generations):
        fitnesses = [fitness(t, dist) for t in population]

        # Cập nhật best
        for t in population:
            L = tour_length(t, dist)
            if L < best_len:
                best_len = L
                best_tour = t[:]

        # Thu thập dữ liệu hội tụ
        if gen <= max_plot_gen and (gen % convergence_interval == 0 or gen == max_plot_gen):
            convergence.append(best_len)
            gen_list.append(gen)

        if verbose and gen % 50 == 0:
            print(f"Gen {gen:4d} - Best length: {best_len:.4f}")

        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitnesses, k=tournament_k)
            p2 = tournament_selection(population, fitnesses, k=tournament_k)

            c1, c2 = (order_crossover(p1, p2) if random.random() < pc else (p1[:], p2[:]))
            c1, c2 = swap_mutation(c1, pm), swap_mutation(c2, pm)

            if random.random() < local_search_rate:
                c1 = two_opt(c1, dist)
            if len(new_pop)+1 < pop_size and random.random() < local_search_rate:
                c2 = two_opt(c2, dist)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

    # ======= HIỂN THỊ KẾT QUẢ =======
    print("\n===== KẾT QUẢ CUỐI CÙNG =====")
    print("Best GA tour:   ", best_tour)
    print("Best GA length: ", best_len)

    # ======= VẼ SƠ ĐỒ HỘI TỤ =======
    plt.figure(figsize=(8,5))
    plt.plot(gen_list, convergence, 'b-', linewidth=2, label="Hybrid GA Best")
    plt.axhline(optimal_len, color='red', linestyle='--', linewidth=2, label="Optimal")
    plt.xlabel("Generation")
    plt.ylabel("Best tour length")
    plt.title(f"Hybrid GA Convergence (0 → {max_plot_gen})")
    plt.grid(True)
    plt.legend()
    plt.show()

    return best_tour, best_len, optimal_len


# ----- Chạy thử -----
if __name__ == "__main__":
    random.seed(42)
    n_cities = 10       # brute-force tốt nhất là n <= 10
    coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_cities)]

    best_tour, best_len, optimal = hybrid_ga_tsp(
        coords,
        pop_size=80,
        generations=400,
        pc=0.9,
        pm=0.2,
        tournament_k=3,
        local_search_rate=1.0,
        verbose=True,
        max_plot_gen=100,
        convergence_interval=5
    )
