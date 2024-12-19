import itertools
from scipy.optimize import minimize

# 更新后的距离和参数
distances = {
    ('O', 1): 50, ('O', -1): 200,
    (1, 2): 50, (-1, -2): 200,
    (1, -2): 150, (-1, 2): 150,
    (2, 'D'): 50, (-2, 'D'): 200,
    (-2, 2): 100000, (2, -2): 100000
}
f = 2.0  # 增加票价
c = 1000
R = 15000  # 增加TOD收益
od_matrix = {
    (1, 2): 600, (1, -2): 100, (1, 'D'): 800,
    (-1, 2): 100, (-1, -2): 100, (-1, 'D'): 100,
    (2, 'D'): 800, (-2, 'D'): 100, (-2, 2): 0,
    (2, -2): 0
}
E = 5e6  # 降低建设入场成本
A = {1: 2500, -1: 5000, 2: 2500, -2: 5000}  # 降低建设面积
a = 5000  # 降低单位面积建设成本
C = 600  # 降低TOD成本
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
    for i, city_i in enumerate(cities):
        if x[city_i] == 1:
            for j, city_j in enumerate(cities):
                if x[city_j] == 1 and city_i != city_j:
                    L_ij = distances.get((city_i, city_j), distances.get((city_j, city_i), 0))
                    q_ij = od_matrix.get((city_i, city_j), 0)
                    fare_revenue += L_ij * q_ij * f
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
def objective(vars):
    y, w, z = vars[:len(cities)], vars[len(cities):2*len(cities)], vars[2*len(cities):]
    y_dict = dict(zip(cities, y))
    w_dict = dict(zip(cities, w))
    z_dict = dict(zip(cities, z))
    profit = calculate_railway_department_profit(x, y_dict, w_dict, z_dict)
    
    # 增加惩罚项，惩罚选择过多城市
    penalty = sum(x.values()) * 1000000  # 根据需要调整惩罚系数
    return -(profit - penalty)
def optimize_decision():
    max_iterations = 100  # 添加最大迭代次数限制
    n = 0
    best_x = None
    best_k = {'y': {city: 0 for city in cities}, 'w': {city: 0 for city in cities}, 'z': {city: 0 for city in cities}}
    max_profit = float('-inf')
    
    while n < max_iterations:
        new_best_x = None
        new_best_k = {'y': {city: 0 for city in cities}, 'w': {city: 0 for city in cities}, 'z': {city: 0 for city in cities}}
        
        for combination in itertools.product([0, 1], repeat=len(cities)):
            x = dict(zip(cities, combination))
            
            # 放宽约束条件：允许多个城市被选择
            if sum(x.values()) < 2:
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
        
        if n > 0 and abs(max_profit - calculate_railway_department_profit(best_x, best_k['y'], best_k['w'], best_k['z'])) < 1e-6:
            break
            
        best_x = new_best_x
        best_k = new_best_k
        n += 1
    
    return best_x, best_k

best_x, best_k = optimize_decision()
print("Best Decision Variables X:", best_x)
print("Optimal Values of Y, W, Z:", best_k)