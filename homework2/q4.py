from scipy.optimize import minimize
import numpy as np
import math
import random

# функция для расчета значений целевой функции


def func(x):
    return -x[0]**2-1.5*x[1]**2+2*x[0]*x[1]-4*x[0]+8*x[1]

# функция вычисления градиента целевой функции


def func_deriv(x):
    dfx0 = -2*x[0]+2*x[1]-4
    dfx1 = -3*x[1]+2*x[0]+8
    return np.array([dfx0, dfx1])


# кортеж для задания ограничений - каждое ограничение - словарь с описанием типа условия, функции вычисления левой части и функкции вычисления градиента левой части
cons = ({'type':'ineq', 'fun': lambda x: np.array([-x[0]-x[1]+3]), 'jac': lambda x: np.array([-1,-1])},
{'type':'ineq', 'fun': lambda x: np.array([-x[0]+x[1]+1]), 'jac': lambda x: np.array([-1,1])},
{'type':'ineq', 'fun': lambda x: np.array([x[0]]), 'jac': lambda x: np.array([1,0])},
{'type':'ineq', 'fun': lambda x: np.array([x[1]]), 'jac': lambda x: np.array([0,1])})


# функция минимизации
res = minimize(func, [0, 3], jac=func_deriv, constraints=cons)
print(res)


# эвристический алгоритм спуска с горы
# проверка попадания точки в множество
def in_set(x):
    return (x[0]+x[1] <= 3) and (x[0]-x[1] <= 1) and (x[0] >= 0) and (x[1] >= 0)

# алгоритм спуска с горы


def minimize1(x):
    x_0 = x
    k = 1
    # радиус окрестности
    i = 0
    # Создание листа с x-ами
    x_list = list()
    fx_min_list = list()
    while i < 1000:
        r1 = np.random.uniform(0, k)
        if np.random.randint(100) % 2 == 1:
            r1 = -r1
        r2 = np.random.uniform(0, k)
        if np.random.randint(100) % 2 == 1:
            r2 = -r2
        x_1 = [x_0[0]+r1, x_0[1]+r2]
        if in_set(x_1):
            x_list.append(x_1)
            i += 1

    for item in x_list:
        if len(fx_min_list) < 100:
            fx_min_list.append(item)
        else:
            fx_min_list.sort(reverse=True)
            if func(item) < func(fx_min_list[0]):
                fx_min_list[0] = item

    x_list = list()

    func_res = list()

    for b in range(100):
        lssss = 0
        while lssss < 50:
            r1 = np.random.uniform(0, 1)
            if np.random.randint(100) % 2 == 1:
                r1 = -r1
            r2 = np.random.uniform(0, 1)
            if np.random.randint(100) % 2 == 1:
                r2 = -r2
            if in_set([fx_min_list[lssss][0] + r1, fx_min_list[len(fx_min_list)-1-lssss][1] + r2]):
                fx_min_list.append([fx_min_list[lssss][0] + r1, fx_min_list[99-lssss][1] + r2])
                func_res.append(func([fx_min_list[lssss][0] + r1, fx_min_list[99-lssss][1] + r2]))
                lssss += 1
        func_res.sort()
        fx_min_list.sort()
        func_res = func_res[0:100]
        fx_min_list = fx_min_list[0:100]

    x_0 = fx_min_list[0]
    f_0 = func_res[0]
    return x_0, f_0


x, f = minimize1([0, 3])
print(x)
print(f)
