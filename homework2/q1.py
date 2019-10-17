from scipy.optimize import minimize
import numpy as np

# функция для расчета значений целевой функции
def func(x):
    return (-x[0]**2-1.5*x[1]**2+2*x[0]*x[1]-4*x[0]+8*x[1])

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
res = minimize(func,[0,3],jac=func_deriv, constraints=cons)
print(res)


# эвристический алгоритм спуска с горы
# проверка попадания точки в множество
def in_set(x):
    return (x[0]+x[1]<=3) and (x[0]-x[1]<=1) and (x[0]>=0) and (x[1]>=0)

# алгоритм спуска с горы
def minimize1(x):
    x_0=x
    f_0 = func(x_0)
    # радиус окрестности
    k=1
    # делаем 100000 проб
    for i in range(100000):
        # генерируем случайную точку из окрестности x_0
        # точка должна принадлежать допустимомум множеству
        while True:
            r1=np.random.uniform(0,k)
            if(np.random.randint(100)%2==1):
                r1=-r1
            r2=np.random.uniform(0,k)
            if(np.random.randint(100)%2==1):
                r2=-r2
            x_1=[x_0[0]+r1, x_0[1]+r2]
            if in_set(x_1):
                break
        f_1=func(x_1)
        # переход в новую точку, если она лучше
        if f_0>f_1:
            print(f_1)
            x_0 = x_1
            print(f_0-f_1)
            if f_0-f_1<0.0000001:
                break
            f_0=f_1
            # изменение величины окрестности
            k=k*0.95
    return x_0, f_0


x,f = minimize1([0,3])
print(x)
print(f)
