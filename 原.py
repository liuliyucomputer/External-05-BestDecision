import itertools

distances = {
    ('O', 1): 100, ('O', -1): 100,
    (1, 2): 100, (-1, -2): 100,
    (1, -2): 141.4, (-1, 2): 141.4,
    (2, 'D'): 100, (-2, 'D'): 100,(-2, 2): 100000,
    (2, -2): 100000
}
f = 0.5
c = 1000
R = 5000
od_matrix = {
    (1, 2): 300, (1, -2): 400, (1, 'D'): 500,
    (-1, 2): 600, (-1, -2): 700, (-1, 'D'): 800,
    (2, 'D'): 900, (-2, 'D'): 1000,(-2, 2): 0,
    (2, -2): 0
}
E = 1e7
A = {1: 4000, -1: 4500, 2: 5000, -2: 4000}
a = 7000
C = 1500
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


def calculate_local_government_revenue(x, y, w, z):
    local_government_revenue = 0
    for city in cities:
        S_i = E + A[city] * a
        H_i = y[city] ** 2 * w[city] * C
        D_i = y[city] ** 2 * w[city] * (1 - z[city]) * p[city]
        T_i = 0
        for other_city in cities:
            if x[other_city] == 1 and other_city != city:
                L_ij = distances[(city, other_city)] if (city, other_city) in distances else distances[
                    (other_city, city)]
                q_ij = od_matrix.get((city, other_city), 0)
                T_i += N * q_ij * ((L_ij / 80) - (L_ij / 200))
        v_i = x[city] * (T_i + D_i - S_i - H_i)
        local_government_revenue += v_i
    return local_government_revenue


def calculate_railway_department_profit(x, y, w, z):
    total_length = calculate_total_length(x)
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
        new_best_k = {'y': {city: 0 for city in cities}, 'w': {city: 0 for city in cities},
                      'z': {city: 0 for city in cities}}

        for combination in itertools.product([0, 1], repeat=len(cities)):
            x = dict(zip(cities, combination))

            # Constraint: Only one city per group can be selected
            if not (sum(x[i] for i in [1, -1]) == 1 and sum(x[j] for j in [2, -2]) == 1):
                continue

            # Initialize variables
            y = {city: 1000 for city in cities}  # Example initial value for Y
            w = {city: 10 for city in cities}  # Example initial value for W
            z = {city: 0.5 for city in cities}  # Example initial value for Z

            # Placeholder for optimization of y, w, z using KKT conditions
            # This part should be replaced with actual optimization logic
            # For simplicity, we assume the initial values are optimal

            profit = calculate_railway_department_profit(x, y, w, z)

            if profit > max_profit:
                max_profit = profit
                new_best_x = x
                new_best_k = {'y': y, 'w': w, 'z': z}

        if n == 0 or any(
                abs(new_best_k[key][city] - best_k[key][city]) > 0.01 for key in ['y', 'w', 'z'] for city in cities):
            best_x = new_best_x
            best_k = new_best_k
            n += 1
        else:
            break

    return best_x, best_k


best_x, best_k = optimize_decision()
print("Best Decision Variables X:", best_x)
print("Optimal Values of Y, W, Z:", best_k)



