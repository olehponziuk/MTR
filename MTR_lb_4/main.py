import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

expected_returns = np.array([0.10, 0.20, 0.50])
std_devs = np.array([0.02, 0.10, 0.20])
correlation_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, -0.6],
    [0.0, -0.6, 1.0]
])

cov_matrix = np.outer(std_devs, std_devs) * correlation_matrix

def portfolio_return(weights):
    return np.dot(weights, expected_returns)

def portfolio_risk(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = [(0, 1) for _ in range(len(expected_returns))]

# Завдання (а): Мінімізація ризику для збереження капіталу
result_a = minimize(portfolio_risk, x0=np.ones(len(expected_returns)) / len(expected_returns), bounds=bounds, constraints=constraints)
weights_a = result_a.x
return_a = portfolio_return(weights_a)
risk_a = portfolio_risk(weights_a)

print("Завдання (а): Збереження капіталу")
print("Ваги портфеля:", weights_a)
print("Очікуваний прибуток:", return_a)
print("Ризик (стандартне відхилення):", risk_a)

# Завдання (б): Досягнення бажаного прибутку 30%
constraints_b = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                 {'type': 'eq', 'fun': lambda x: portfolio_return(x) - 0.30}]
result_b = minimize(portfolio_risk, x0=np.ones(len(expected_returns)) / len(expected_returns), bounds=bounds, constraints=constraints_b)
weights_b = result_b.x
return_b = portfolio_return(weights_b)
risk_b = portfolio_risk(weights_b)

print("\nЗавдання (б): Досягнення бажаного прибутку 30%")
print("Ваги портфеля:", weights_b)
print("Очікуваний прибуток:", return_b)
print("Ризик (стандартне відхилення):", risk_b)

# Завдання (в): Приріст капіталу при ризику 15%
constraints_c = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                 {'type': 'eq', 'fun': lambda x: portfolio_risk(x) - 0.15}]
result_c = minimize(lambda x: -portfolio_return(x), x0=np.ones(len(expected_returns)) / len(expected_returns), bounds=bounds, constraints=constraints_c)
weights_c = result_c.x
return_c = portfolio_return(weights_c)
risk_c = portfolio_risk(weights_c)

print("\nЗавдання (в): Приріст капіталу при ризику 15%")
print("Ваги портфеля:", weights_c)
print("Очікуваний прибуток:", return_c)
print("Ризик (стандартне відхилення):", risk_c)

# Завдання (д): Побудова ефективного кордону
target_returns = np.linspace(0.10, 0.50, 100)
efficient_risks = []

for target_return in target_returns:
    constraints_d = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                     {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return}]
    result_d = minimize(portfolio_risk, x0=np.ones(len(expected_returns)) / len(expected_returns), bounds=bounds, constraints=constraints_d)
    efficient_risks.append(portfolio_risk(result_d.x))

# Побудова графіку
plt.figure(figsize=(10, 6))
plt.plot(efficient_risks, target_returns, label="Ефективний кордон")
plt.scatter([risk_a, risk_b, risk_c], [return_a, return_b, return_c], color='red', marker='o', label="Рішення задач (а), (б), (в)")
plt.xlabel("Ризик (середньоквадратичне відхилення)")
plt.ylabel("Очікуваний прибуток")
plt.title("Ефективний кордон портфеля")
plt.legend()
plt.grid(True)
plt.show()
