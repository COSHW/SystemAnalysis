import numpy as np
from scipy.optimize import linprog

m = 3
n = 3
c = [3, 6, 5, 4, 9, 2, 3, 6, 2]
a = [58, 60, 40]
b = [30, 50, 20]

A_1 = [[1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1]]
A_2 = [[1, 0, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1]]

res = linprog(c, A_1, a, A_2, b)
print(res)
print("----------------------------------------------------")

# изменения цен перевозки - коэффициентов целевой функции
c_new = [7, 5, 6, 4, 8, 2, 7, 5, 9]

# объединение матриц ограничений неравенств и равенств для получения полных столюцов p_j
A = np.array([A_1, A_2])
A = np.reshape(A, [m + n, m * n])


# объединение коэффициентов правой части
B = np.array([a, b])
B.resize(m + n)


# формирование матрицы единичных векторов для дополнительных переменных
A_slack = np.zeros([m + n, m])
for i in range(m):
    A_slack[i, i] = 1


# получение номеров базисных векторов по решению задачи (ненулевые компоненты) - основные вектора
base_ind = np.nonzero(res.x)[0]

# получение номеров базисных векторов по решению задачи (ненулевые компоненты) - дополнительные вектора
# требуется корректировка номеров векторов на количество основных векторов задачи (+m*n)
base_ind_dop = np.nonzero(res.slack)[0] + m * n

# объединение списка номеров базисных переменных
base_ind = np.concatenate((base_ind, base_ind_dop))

# формирование базисной матрицы и столбца базисных коэффициентов целевой функции
basis = []
c_bas = []
for i in range(m + n):
    # если вектор основной
    if base_ind[i] < m * n:
        basis.append(A[:, base_ind[i]])
        c_bas.append(c_new[base_ind[i]])
    else:
        # если вектор дополнительный
        basis.append(A_slack[:, base_ind[i] - m * n])
        c_bas.append(0)

# вектора добавляются в матрицу как вектора-строки. Нужно - вектора-столбцы
# проведение транспонирвания
basis = np.reshape(basis, (m + n, m + n)).T

# получение обратной матрицы
basis_1 = np.linalg.inv(basis)

while True:
    # получение вектора cb*B(-1) для дальнейшего получения оценок
    cb = np.dot(c_bas, basis_1)
    # получение оценок основных векторов
    delta = np.dot(cb, A) - c_new
    # получение оценов дополнительных векторов
    delta_dop = np.dot(cb, A_slack)
    # получение коэффициентов разложения вектора p_0
    p_0 = np.dot(basis_1, B)

    # получение максимальной оценки
    max_delta = np.max(delta)
    max_delta_dop = np.max(delta_dop)
    # если максимальная оценка равна 0, получен оптимум
    if max(max_delta, max_delta_dop) == 0:
        print("optimum")
        break
    else:
        # если положительная оценка у основного вектора  
        if max_delta > 0:
            # запоминаем индекс основного вектора
            ind_to_basis = np.argmax(delta)
        else:
            # запоминаем индекс дополнительного вектора - корректируем номера векторов
            ind_to_basis = np.argmax(delta_dop) + m * n

    # вычисляем коэффициенты разложения по базису вектора с положительно оценкой    
    if ind_to_basis < m * n:
        p_j = np.dot(basis_1, A[:, ind_to_basis])
        c_new_basis = c_new[ind_to_basis]
    else:
        p_j = np.dot(basis_1, A_slack[:, ind_to_basis - m * n])
        c_new_basis = 0

    # находим в вектор, который выводится из базиса 
    ind = -1
    min = 100000
    for i in range(m + n):
        if p_j[i] > 0:
            if p_0[i] / p_j[i] < min:
                ind = i
                min = p_0[i] / p_j[i]

    # осуществляем пересчет по формулам Гаусса
    # замена номера базисного вектора
    base_ind[ind] = ind_to_basis
    # замена коэффициента целевой функции при базисном векторе
    c_bas[ind] = c_new_basis
    # пересчет
    basis_1[ind, :] = basis_1[ind, :] / p_j[ind]
    for i in range(m + n):
        if i != ind:
            basis_1[i, :] = basis_1[i, :] - basis_1[ind, :] * p_j[i]

# печать коэффициентов разложения p_0, из которого можно получить ответ
print("----------------------------------------------------")
print(p_0)
print(base_ind)
