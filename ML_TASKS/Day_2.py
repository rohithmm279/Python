import matplotlib.pyplot as plt

# -----------------------------
# DATA
# -----------------------------
hours_studied = [1, 2, 3, 5, 6, 8]
actual_marks = [10, 19, 28, 38, 51, 60]
# =====================================================
# TASK 2 : Weight + Bias
# =====================================================

new_students = [4, 7, 9]

best_weight = None
best_bias = None
lowest_mse = float('inf')

for weight in range(1, 21):
    for bias in range(-10, 11):

        predictions = [weight * h + bias for h in hours_studied]

        mse = sum((p - a) ** 2 for p, a in zip(predictions, actual_marks)) / len(actual_marks)

        if mse < lowest_mse:
            lowest_mse = mse
            best_weight = weight
            best_bias = bias

print("\nTASK 2")
print("Best Weight:", best_weight)
print("Best Bias:", best_bias)
print("Lowest MSE:", lowest_mse)

# Predictions for new students
print("\nPredictions:")
for hrs in new_students:
    predicted_marks = best_weight * hrs + best_bias
    print(f"Student studied {hrs} hrs → predicted marks: {predicted_marks}")