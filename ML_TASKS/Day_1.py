import matplotlib.pyplot as plt

# -----------------------------
# DATA
# -----------------------------
hours_studied = [1, 2, 3, 5, 6, 8]
actual_marks = [10, 19, 28, 38, 51, 60]

# =====================================================
# TASK 1 : Find Best Weight
# =====================================================

weights = range(1, 21)
mse_values = []

best_weight = None
lowest_mse = float('inf')

for weight in weights:
    predictions = [weight * h for h in hours_studied]

    mse = sum((p - a) ** 2 for p, a in zip(predictions, actual_marks)) / len(actual_marks)

    mse_values.append(mse)

    if mse < lowest_mse:
        lowest_mse = mse
        best_weight = weight

print("TASK 1")
print("Best Weight:", best_weight)
print("Lowest MSE:", lowest_mse)

# Plot Weight vs MSE
plt.figure(figsize=(8, 5))
plt.plot(weights, mse_values, marker='o')
plt.title("Weight vs MSE")
plt.xlabel("Weight")
plt.ylabel("MSE")
plt.grid(True)
plt.show()
