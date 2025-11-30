import random
import math
import matplotlib.pyplot as plt

# ===================================================
#                BASIC FUNCTIONS
# ===================================================
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
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += dist[tour[i]][tour[(i+1)%n]]
    return total

def fitness(tour, dist):
    L = tour_length(tour, dist)
    return 1.0 / L

def random_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour

# ===================================================
#                 GA OPERATORS
# ===================================================
def tournament_selection(pop, fits, k=3):
    best_idx = None
    for _ in range(k):
        idx = random.randrange(len(pop))
        if best_idx is None or fits[idx] > fits[best_idx]:
            best_idx = idx
    return pop[best_idx][:]

def order_crossover(p1, p2):
    n = len(p1)
    c1, c2 = [None]*n, [None]*n
    i, j = sorted(random.sample(range(n), 2))
    c1[i:j+1] = p1[i:j+1]
    c2[i:j+1] = p2[i:j+1]

    def fill(child, parent):
        pos = (j+1) % n
        for gene in parent:
            if gene not in child:
                child[pos] = gene
                pos = (pos + 1) % n

    fill(c1, p2)
    fill(c2, p1)
    return c1, c2

def swap_mutation(tour, pm):
    if random.random() < pm:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

# ===================================================
#                2-OPT LOCAL SEARCH
# ===================================================
def two_opt(tour, dist):
    best = tour[:]
    best_len = tour_length(best, dist)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best)-2):
            for j in range(i+1, len(best)-1):
                new_tour = best[:]
                new_tour[i:j+1] = reversed(new_tour[i:j+1])
                new_len = tour_length(new_tour, dist)

                if new_len < best_len:
                    best = new_tour
                    best_len = new_len
                    improved = True

    return best

# ===================================================
#                HYBRID GA + 2-OPT
# ===================================================
def hybrid_ga(coords, pop_size=80, generations=300, pc=0.9, pm=0.2,
              tournament_k=3, local_search_rate=1.0,
              max_plot_gen=200, interval=5):

    n = len(coords)
    dist = distance_matrix(coords)

    population = [random_tour(n) for _ in range(pop_size)]
    best_len = float("inf")
    best_tour = None

    convergence, gen_list = [], []

    for gen in range(generations):
        fits = [fitness(t, dist) for t in population]

        # update global best
        for t in population:
            L = tour_length(t, dist)
            if L < best_len:
                best_len = L
                best_tour = t[:]

        # save convergence
        if gen <= max_plot_gen and gen % interval == 0:
            convergence.append(best_len)
            gen_list.append(gen)

        # create new population
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fits, tournament_k)
            p2 = tournament_selection(population, fits, tournament_k)

            if random.random() < pc:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            c1 = swap_mutation(c1, pm)
            c2 = swap_mutation(c2, pm)

            if random.random() < local_search_rate:
                c1 = two_opt(c1, dist)
            if random.random() < local_search_rate:
                c2 = two_opt(c2, dist)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

    return best_tour, best_len, gen_list, convergence

# ===================================================
#                STANDARD GA (NO 2-OPT)
# ===================================================
def standard_ga(coords, pop_size=80, generations=300, pc=0.9, pm=0.2,
                tournament_k=3, max_plot_gen=200, interval=5):

    n = len(coords)
    dist = distance_matrix(coords)

    population = [random_tour(n) for _ in range(pop_size)]
    best_len = float("inf")
    best_tour = None

    convergence, gen_list = [], []

    for gen in range(generations):
        fits = [fitness(t, dist) for t in population]

        for t in population:
            L = tour_length(t, dist)
            if L < best_len:
                best_len = L
                best_tour = t[:]

        if gen <= max_plot_gen and gen % interval == 0:
            convergence.append(best_len)
            gen_list.append(gen)

        new_pop = []

        # elitism
        best_idx = max(range(pop_size), key=lambda i: fits[i])
        new_pop.append(population[best_idx][:])

        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fits, tournament_k)
            p2 = tournament_selection(population, fits, tournament_k)

            if random.random() < pc:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            c1 = swap_mutation(c1, pm)
            c2 = swap_mutation(c2, pm)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

    return best_tour, best_len, gen_list, convergence

# ===================================================
#                   RUN BOTH + PLOT
# ===================================================
if __name__ == "__main__":
    random.seed(42)

    n_cities = 15
    coords = [(random.uniform(0,100), random.uniform(0,100)) for _ in range(n_cities)]

    # Run both
    h_tour, h_len, h_gens, h_conv = hybrid_ga(coords)
    s_tour, s_len, s_gens, s_conv = standard_ga(coords)

    print("Hybrid GA best length:  ", h_len)
    print("Standard GA best length:", s_len)

    # Plot comparison
    plt.figure(figsize=(10,6))
    plt.plot(h_gens, h_conv, label="Hybrid GA (with 2-opt)", linewidth=2)
    plt.plot(s_gens, s_conv, label="Standard GA (no 2-opt)", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Best tour length")
    plt.title("Convergence Comparison")
    plt.grid(True)
    plt.legend()
    plt.show()
