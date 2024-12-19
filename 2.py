import itertools
from scipy.optimize import minimize

# 定义距离和其他参数
distances = {
    ('O', 1): 50, ('O', -1): 200,
    (1, 2): 50, (-1, -2): 200,
    (1, -2): 150, (-1, 2): 150,
    (2, 'D'): 50, (-2, 'D'): 200,
    (-2, 2): 100000, (2, -2): 100000
}
f = 1.0
c = 1000
R = 10000
od_matrix = {
    (1, 2): 500, (1, -2): 100, (1, 'D'): 700,
    (-1, 2): 100, (-1, -2): 100, (-1, 'D'): 100,
    (2, 'D'): 700, (-2, 'D'): 100, (-2, 2): 0,
    (2, -2): 0
}
E = 1e7
A = {1: 3000, -1: 6000, 2: 3000, -2: 6000}
a = 7000
C = 800
g = {1: 1e13, -1: 8e12, 2: 7.5e12, -2: 9e12}
p = {1: 10000, -1: 9000, 2: 8000, -2: 10000}
N = 30
cities = [1, -1, 2, -2]

def calculate_total_length(x):
    total_length = 0
    prev_city = 'O'
    for city in cities:
        if x[city] == 1:
            total_length += distances[(prev_city, city)]
            prev_city = city
    total_length += distances[(prev_city, 'D')]
    return total_length

def calculate_fare_revenue(x):
    fare_revenue = 0
    prev_city = 'O'
    for i, city_i in enumerate(cities):
        if x[city_i] == 1:
            for j, city_j in enumerate(cities[i+1:], start=i+1):
                if x[city_j] == 1:
                    L_ij = sum(distances[(cities[k], cities[k+1])] for k in range(i, j) if (cities[k], cities[k+1]) in distances or (cities[k+1], cities[k]) in distances)
                    q_ij = od_matrix.get((city_i, city_j), 0)
                    fare_revenue += L_ij * q_ij * f
            q_id = od_matrix.get((city_i, 'D'), 0)
            L_id = sum(distances[(cities[k], cities[k+1])] for k in range(i, len(cities)-1) if (cities[k], cities[k+1]) in distances or (cities[k+1], cities[k]) in distances)
            fare_revenue += L_id * q_id * f
            break
    else:
        if x[cities[-1]] == 1:
            q_id = od_matrix.get((cities[-1], 'D'), 0)
            fare_revenue += distances[(cities[-1], 'D')] * q_id * f
    return fare_revenue

def calculate_tod_revenue(x, y, w, z):
    tod_revenue = 0
    for city in cities:
        if x[city] == 1:
            tod_revenue += y[city] ** 2 * w[city] * z[city] * R
    return tod_revenue

def calculate_tod_cost(x, y, w):
    tod_cost = 0
    for city in cities:
        if x[city] == 1:
            tod_cost += y[city] ** 2 * w[city] * C
    return tod_cost

def calculate_station_cost(x):
    station_cost = 0
    for city in cities:
        if x[city] == 1:
            station_cost += E + A[city] * a
    return station_cost

def calculate_railway_department_profit(x, y, w, z):
    fare_revenue = calculate_fare_revenue(x)
    tod_revenue = calculate_tod_revenue(x, y, w, z)
    tod_cost = calculate_tod_cost(x, y, w)
    station_cost = calculate_station_cost(x)
    profit = fare_revenue + tod_revenue - tod_cost - station_cost
    return profit

def optimize_decision():
    n = 0
    best_x = None
    best_k = {'y': {city: 0 for city in cities}, 'w': {city: 0 for city in cities}, 'z': {city: 0 for city in cities}}
    max_profit = float('-inf')
    
    while True:
        new_best_x = None
        new_best_k = {'y': {city: 0 for city in cities}, 'w': {city: 0 for city in cities}, 'z': {city: 0 for city in cities}}
        
        for combination in itertools.product([0, 1], repeat=len(cities)):
            x = dict(zip(cities, combination))
            
            # 放宽约束条件：允许多个城市被选择
            if sum(x.values()) == 0:
                continue
            
            # 打印当前组合
            print(f"Testing combination: {x}")
            
            # 优化 y, w, z
            def objective(vars):
                y, w, z = vars[:len(cities)], vars[len(cities):2*len(cities)], vars[2*len(cities):]
                y_dict = dict(zip(cities, y))
                w_dict = dict(zip(cities, w))
                z_dict = dict(zip(cities, z))
                profit = calculate_railway_department_profit(x, y_dict, w_dict, z_dict)
                # 打印当前解的收益
                print(f"Current profit: {profit}, y: {y_dict}, w: {w_dict}, z: {z_dict}")
                return -profit
            
            # 初始值和边界
            initial_values = [1500] * len(cities) + [20] * len(cities) + [0.5] * len(cities)
            bounds = [(500, 2500)] * len(cities) + [(10, 30)] * len(cities) + [(0.1, 0.8)] * len(cities)
            
            result = minimize(objective, initial_values, bounds=bounds, method='SLSQP')
            
            if result.success:
                y_opt, w_opt, z_opt = result.x[:len(cities)], result.x[len(cities):2*len(cities)], result.x[2*len(cities):]
                y_dict = dict(zip(cities, y_opt))
                w_dict = dict(zip(cities, w_opt))
                z_dict = dict(zip(cities, z_opt))
                profit = calculate_railway_department_profit(x, y_dict, w_dict, z_dict)
                
                if profit > max_profit:
                    max_profit = profit
                    new_best_x = x
                    new_best_k = {'y': y_dict, 'w': w_dict, 'z': z_dict}
        
        if n == 0 or any(abs(new_best_k[key][city] - best_k[key][city]) > 0.01 for key in ['y', 'w', 'z'] for city in cities):
            best_x = new_best_x
            best_k = new_best_k
            n += 1
        else:
            break
    
    return best_x, best_k

best_x, best_k = optimize_decision()
print("Best Decision Variables X:", best_x)
print("Optimal Values of Y, W, Z:", best_k)