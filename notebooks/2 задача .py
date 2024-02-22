import numpy as np
from scipy.optimize import linprog

# Приведение задачи к стандартной форме
c = [-1, -1, 0, 0]  # коэффициенты целевой функции (минимизируем)
A_eq = [[1, -1, -1, 0], [1, -2, 0, -1]]  # коэффициенты при переменных в уравнениях
b_eq = [1, 2]  # правая часть уравнений
bounds = [(0, None), (0, None), (0, None), (0, None)]  # границы переменных

# Преобразование A_eq к двумерному массиву
A_eq = np.array(A_eq)

# Решение задачи линейного программирования
res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

print("Решение задачи ЛП:")
print(res)

# Найдем все базовые допустимые решения
basic_feasible_solutions = []

for i in range(4):
    for j in range(4):
        if i != j:
            x = [0, 0, 0, 0]
            x[i] = 0
            x[j] = 0
            if A_eq[0][i] != 0:
                x[2] = b_eq[0] / A_eq[0][i]
                x[j] = b_eq[0] / A_eq[0][i]
            elif A_eq[0][j] != 0:
                x[2] = b_eq[0] / A_eq[0][j]
                x[i] = b_eq[0] / A_eq[0][j]

            if A_eq[1][i] != 0:
                x[3] = b_eq[1] / A_eq[1][i]
                x[j] = b_eq[1] / A_eq[1][i]
            elif A_eq[1][j] != 0:
                x[3] = b_eq[1] / A_eq[1][j]
                x[i] = b_eq[1] / A_eq[1][j]

            if all(x[k] >= 0 for k in range(4)):
                basic_feasible_solutions.append(x)

print("\nВсе базовые допустимые решения:")
for bfs in basic_feasible_solutions:
    print(bfs)

# Найдем решение, максимизирующее целевую функцию
min_z = float('inf')
optimal_solution = None
for bfs in basic_feasible_solutions:
    z = np.dot(c, bfs)  # значение целевой функции
    if z < min_z:
        min_z = z
        optimal_solution = bfs

print("\nРешение, максимизирующее целевую функцию:")
print("Значение целевой функции:", min_z)
print("Оптимальное решение:", optimal_solution[:2])
