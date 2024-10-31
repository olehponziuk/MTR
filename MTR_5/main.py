import numpy as np

def fishburn_second_formula(n):
    probabilities = []
    for j in range(1, n + 1):
        p_j = 2 ** (n - j) / (2 ** n - 1)
        probabilities.append(p_j)
    return probabilities

F = np.array([
    [3, 6, 5, 6],
    [1, 3, 9, 5],
    [4, 1, 8, 4]
])

n = F.shape[1]
probabilities = fishburn_second_formula(n)

expected_profits = F.dot(probabilities)

optimal_decision_index = np.argmax(expected_profits)
optimal_decision = expected_profits[optimal_decision_index]

print("Ймовірності за другою формулою Фішберна:", probabilities)
print("Очікуваний прибуток для кожного варіанту рішення:", expected_profits)
print("Оптимальне рішення: x_", optimal_decision_index + 1, "з очікуваним прибутком:", optimal_decision)
