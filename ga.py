import numpy as np
import matplotlib.pyplot as plt

# --- 1. HÀM MỤC TIÊU (FITNESS FUNCTION) ---
def fitness_function(x):
    """
    Hàm mục tiêu: f(x) = x * sin(10*pi*x) + 1
    Mục tiêu: Cực đại hóa
    """
    return x * np.sin(10 * np.pi * x) + 1

# --- 2. THAM SỐ CƠ BẢN CỦA GA ---
POPULATION_SIZE = 50      # Kích thước quần thể
GENERATIONS = 100         # Số thế hệ tối đa
LOWER_BOUND = 0.0         # Giới hạn dưới của miền giá trị x
UPPER_BOUND = 1.0         # Giới hạn trên của miền giá trị x
CROSSOVER_RATE = 0.8      # Tỷ lệ lai ghép (Pc)
MUTATION_RATE = 0.1       # Tỷ lệ đột biến (Pm)

# --- 3. KHỞI TẠO QUẦN THỂ ---
def initialize_population(size, lower, upper):
    """
    Khởi tạo ngẫu nhiên quần thể các giá trị x trong khoảng [lower, upper]
    """
    return np.random.uniform(lower, upper, size)

# --- 4. CHỌN LỌC (SELECTION: TOURNAMENT SELECTION) ---
def tournament_selection(population, fitness_scores, k=3):
    """
    Chọn lọc Tournament với kích thước k=3
    """
    new_population = np.empty_like(population)
    
    for i in range(len(population)):
        # Chọn ngẫu nhiên k cá thể cho tournament
        indices = np.random.choice(len(population), k, replace=False)
        tournament_participants = fitness_scores[indices]
        
        # Chọn cá thể có fitness tốt nhất
        winner_index = indices[np.argmax(tournament_participants)]
        new_population[i] = population[winner_index]
        
    return new_population

# --- 5. LAI GHÉP (CROSSOVER: UNIFORM ARITHMETIC CROSSOVER) ---
def arithmetic_crossover(parent1, parent2, rate):
    """
    Lai ghép số học: child = alpha * parent1 + (1 - alpha) * parent2
    """
    if np.random.rand() < rate:
        # alpha được chọn ngẫu nhiên trong [0, 1]
        alpha = np.random.rand()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1 # Tạo thêm child2 để thay thế
        return child1, child2
    else:
        return parent1, parent2 # Không lai ghép, giữ nguyên bố mẹ

def crossover(population, rate, lower, upper):
    """
    Thực hiện lai ghép cho toàn bộ quần thể được chọn
    """
    next_population = np.empty_like(population)
    
    # Đảm bảo vòng lặp theo cặp
    for i in range(0, len(population) - 1, 2):
        parent1 = population[i]
        parent2 = population[i+1]
        
        child1, child2 = arithmetic_crossover(parent1, parent2, rate)
        
        # Đảm bảo giá trị nằm trong giới hạn [0, 1]
        next_population[i] = np.clip(child1, lower, upper)
        next_population[i+1] = np.clip(child2, lower, upper)
        
    # Xử lý cá thể cuối cùng nếu số lượng lẻ
    if len(population) % 2 != 0:
        next_population[-1] = population[-1] 
        
    return next_population

# --- 6. ĐỘT BIẾN (MUTATION: NON-UNIFORM MUTATION) ---
def non_uniform_mutation(x, gen, max_gen, rate, lower, upper):
    """
    Đột biến không đồng nhất: Độ lớn của sự thay đổi giảm dần theo số thế hệ (gen)
    """
    if np.random.rand() < rate:
        # Hệ số điều chỉnh độ lớn của đột biến (b)
        # Sử dụng b=2 để độ lớn giảm nhanh
        b = 2 
        
        # Hàm xác suất thay đổi p(t)
        r = np.random.rand()
        p_t = (1 - gen / max_gen)**b
        
        # Quyết định hướng thay đổi (delta)
        if np.random.rand() < 0.5: # Hướng dương
            delta = (upper - x) * (1 - r**(p_t))
            return x + delta
        else: # Hướng âm
            delta = (x - lower) * (1 - r**(p_t))
            return x - delta
    else:
        return x

def mutate(population, gen, max_gen, rate, lower, upper):
    """
    Thực hiện đột biến cho toàn bộ quần thể
    """
    mutated_population = np.empty_like(population)
    for i, x in enumerate(population):
        mutated_population[i] = non_uniform_mutation(x, gen, max_gen, rate, lower, upper)
        
    # Đảm bảo giá trị vẫn nằm trong giới hạn [0, 1]
    return np.clip(mutated_population, lower, upper)

# --- 7. CHẠY THUẬT TOÁN DI TRUYỀN (GA MAIN LOOP) ---
def run_ga():
    
    # Khởi tạo
    population = initialize_population(POPULATION_SIZE, LOWER_BOUND, UPPER_BOUND)
    best_fitness_history = []
    avg_fitness_history = []
    
    best_solution = None
    best_fitness = -np.inf

    print("Bắt đầu mô phỏng Thuật toán Di truyền...")
    print(f"Số thế hệ: {GENERATIONS}, Kích thước quần thể: {POPULATION_SIZE}")

    for gen in range(GENERATIONS):
        
        # 1. Đánh giá độ thích nghi
        fitness_scores = fitness_function(population)
        
        # Lưu trữ lịch sử kết quả
        current_best_fitness = np.max(fitness_scores)
        current_avg_fitness = np.mean(fitness_scores)
        
        best_fitness_history.append(current_best_fitness)
        avg_fitness_history.append(current_avg_fitness)
        
        # Cập nhật giải pháp tốt nhất (Elitism)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[np.argmax(fitness_scores)]
        
        # 2. Chọn lọc
        selected_population = tournament_selection(population, fitness_scores)
        
        # 3. Lai ghép
        crossover_population = crossover(selected_population, CROSSOVER_RATE, LOWER_BOUND, UPPER_BOUND)
        
        # 4. Đột biến
        mutated_population = mutate(crossover_population, gen, GENERATIONS, MUTATION_RATE, LOWER_BOUND, UPPER_BOUND)
        
        # 5. Tạo thế hệ mới
        population = mutated_population
        
        # Hiển thị tiến trình (mỗi 10 thế hệ)
        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"Thế hệ {gen+1}/{GENERATIONS}: Max F(x) = {current_best_fitness:.4f}, x = {best_solution:.4f}")

    print("\nMô phỏng hoàn tất.")
    print(f"Giá trị tối ưu toàn cục (x) tìm được: {best_solution:.4f}")
    print(f"Giá trị cực đại của hàm F(x): {best_fitness:.4f}")

    return best_fitness_history, avg_fitness_history, best_solution, best_fitness

# --- 8. VẼ SƠ ĐỒ HỘI TỤ (CONVERGENCE PLOT) ---
def plot_convergence(best_history, avg_history, best_x, best_f):
    
    plt.figure(figsize=(10, 6))
    plt.plot(best_history, label='Độ Thích Nghi Tốt Nhất (Best Fitness)', color='green', linewidth=2)
    plt.plot(avg_history, label='Độ Thích Nghi Trung Bình (Average Fitness)', color='orange', linestyle='--')
    
    plt.title(f'Sơ Đồ Hội Tụ của Thuật toán Di truyền (GA)', fontsize=14)
    plt.xlabel('Số Thế Hệ (Generation)', fontsize=12)
    plt.ylabel('Giá Trị Hàm Mục Tiêu F(x)', fontsize=12)
    
    # Đánh dấu kết quả cuối cùng
    plt.axhline(best_f, color='red', linestyle=':', label=f'F(x) Max = {best_f:.4f} tại x = {best_x:.4f}')
    
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
    

# --- CHẠY CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    
    # 1. Chạy GA
    best_history, avg_history, final_x, final_f = run_ga()
    
    # 2. Vẽ sơ đồ
    plot_convergence(best_history, avg_history, final_x, final_f)