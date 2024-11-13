import numpy as np

F = np.array([
    [250, 350, 150],
    [750, 200, 350],
    [255, 850, 350],
    [800, 550, 450]
])

P = np.array([0.1, 0.5, 0.4])

lambdas = [0.1, 0.3]

expected_profits = F @ P

max_profits_per_condition = np.max(F, axis=0)
opportunity_losses = max_profits_per_condition - F
expected_losses = opportunity_losses @ P

#Застосування критерію Гурвіца
hurwicz_scores = []
for lambda_val in lambdas:
    hurwicz_for_lambda = []
    for row in F:
        max_profit = np.max(row)
        min_profit = np.min(row)
        score = lambda_val * max_profit + (1 - lambda_val) * min_profit
        hurwicz_for_lambda.append(score)
    hurwicz_scores.append(hurwicz_for_lambda)

# Компроміс
compromise_scores = [
    (expected_losses[i] + np.mean([hurwicz_scores[0][i], hurwicz_scores[1][i]])) / 2
    for i in range(len(F))
]

best_decision_index = np.argmin(compromise_scores)
best_decision = best_decision_index + 1

print("Очікувані прибутки:", expected_profits)
print("Очікувані втрати:", expected_losses)
print("Оцінки за Гурвіцем для кожного рішення (λ=0.1 та λ=0.3):", hurwicz_scores)
print("Компромісні оцінки:", compromise_scores)
print(f"Найкраще рішення - варіант {best_decision} з компромісною оцінкою {compromise_scores[best_decision_index]:.2f}")
