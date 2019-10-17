m=3
n=2
c=[-1,-2]
a=[4,4,4]

A_1=[[5,-2],[-1,2],[1,1]]

#x0_bounds = [3,1,4]
#x1_bounds = [100,3,10]

from scipy.optimize import linprog

res=linprog(c, A_ub=A_1,b_ub=a, A_eq=None, b_eq=None, bounds=None, method='simplex', options={"disp": 'false'})
"""list(zip(x0_bounds, x1_bounds))"""
print(res)

# ПЕРВАЯ ЗАДАЧА НА ЧУВСТВИТЕЛЬНОСТЬ
# ИЗМЕНЕНИЕ ПРАВОЙ ЧАСТИ ОГРАНИЧЕНИЙ

a_new = [4,8,2]

import numpy as np

A_1 = np.array(A_1)

# формирование матрицы единичных векторов для дополнительных переменных
A_slack=np.zeros([m,m+n])
for i in range(m):
    A_slack[i, i+n] = 1
print(A_slack)


base_ind =np.nonzero(res.x)[0]
print('X:')
print(base_ind)

base_ind_dop = []
index=n-1
for x in res.slack:
    index=index+1
    if x > 0.001:
        print('Условие выполнено')
        print(index)
        base_ind_dop.append(index)

print('Dop:')
print(base_ind_dop)
base_ind=np.concatenate((base_ind,base_ind_dop))

print('A:')
print(A_1)

print('basis:')
print(base_ind)

basis=[]
c_bas=[]
for i in range(len(base_ind)):
    # если вектор основной
    if base_ind[i]<n: #if base_ind[i]<m*n:
        basis.append(A_1[:, base_ind[i]])
        c_bas.append(c[base_ind[i]])
    else:
        # если вектор дополнительный
        basis.append(A_slack[:,base_ind[i]])  # -m*n
        c_bas.append(0)

print('c_bas: ')
print(c_bas)

#basis = np.matrix(basis)
#print(basis)
# вектора добавляются в матрицу как вектора-строки. Нужно - вектора-столбцы
# проведение транспонирвания
basis = np.array(basis)
basis = basis.transpose()

print('транспонированная:')
print(basis)

# получение обратной матрицы
basis_1=np.linalg.inv(basis)

print('обратная:')
print(basis_1)

#разложение нового p0 по базису
p0 = basis_1.dot(a_new)

print('p0:')
print(p0)

vect_out=-1

for i in range(len(p0)):

    if p0[i]<0:
        print('Двойственный симпл метод')
        vect_out=i

    if vect_out==-1:
        print('Новое условие оптимальности: ')
        print(p0)
    else:
        #Двойственный симпл метод, выводим из базиса
        print('Выводим вектор ')
        print(vect_out)

        #повтор с другими отрицательными

        while True:
            # получение вектора cb*B(-1) для дальнейшего получения оценок
            cb = np.dot(c_bas, basis_1)
            # получение оценок основных векторов
            delta = np.dot(cb, A_1) - c
            # получение оценов дополнительных векторов
            delta_dop = np.dot(cb, A_slack)
            # получение коэффициентов разложения вектора p_0
            p_0 = np.dot(basis_1, a_new)

            #трассировка
            print('Симпл таблица:')
            print(cb)
            print(delta)
            if (delta[0] > 0.001):
                print('true')
            print(delta_dop)
            print(p_0)

            #объединяем оценки
            delta_all = np.concatenate((delta, delta_dop),axis=0)
            print('Все оценки:')
            print(delta_all)

            #ищем вектор коэффициентов разложения (соответствующих выводимому) векторов, не входящих в базис
            koefs = []
            for i in range(m+n):
                if i not in base_ind:
                    koefs.append(basis_1[vect_out, i-n])
             #   if i in base_ind:
             #       koefs = koefs + [0]

            print('Коэффициенты для счёта:')
            print(koefs)

            #ищем вектор оценок при векторах, не входящих в базис
            delta_new = []
            for i in range(len(delta_dop)):
                if i not in base_ind:
                    delta_new.append(delta_dop[i])
            print('оценки для счёта:')
            print(delta_new)

            #ищем вектор для ввода в базис
            #начальное значение выбирается произвольно
            ind_to_basis = (vect_out + 1) % len(delta_new)
            for i in range(len(delta_new)):
                if (abs(delta_new[i]) / abs(koefs[i]) < abs(delta_new[ind_to_basis]) / abs(koefs[ind_to_basis])):
                    ind_to_basis = i

            #возвращаем индекс вводимого вектора среди всех векторов
            ind_to_basis = ind_to_basis + m
            print('Вводим вектор:')
            print(ind_to_basis)

            # вычисляем коэффициенты разложения по базису вводимого вектора
            if ind_to_basis < n:
                p_j = np.dot(basis_1, A_1[:, ind_to_basis])
                c_new_basis = c[ind_to_basis]
            else:
                p_j = np.dot(basis_1, A_slack[:, ind_to_basis])
                c_new_basis = 0

            print(p_j)
            print(c_new_basis)

            # формируем вектор cb
            base_ind[vect_out] = ind_to_basis
            if ind_to_basis<n:
                c_bas[vect_out] = c[ind_to_basis]
            else:
                c_bas[vect_out]=0

            print(c_bas)

            # пересчет
            basis_1[vect_out, :] = basis_1[vect_out, :] / p_j[vect_out]
            for i in range(m):
                if i != vect_out:
                    basis_1[i, :] = basis_1[i, :] - basis_1[vect_out, :] * p_j[i]


            print(basis_1)

            print('_____________ОТВЕТ_____________')
            print('Значения:')
            print(p_0)
            print('Базисные вектора:')
            print(base_ind)
            print('Значение ц/ф:')

            s = 0
            for i in range(len(c_bas)):
                if base_ind[i] < n:
                    s = s + p_0[base_ind[i]]*abs(c_bas[base_ind[i]])

            print(s)

            print('Вывод:')
            print('Решение оптимально.')
            if (s>abs(res.fun)):
                print('Изменения позитивно отразились на целевой функции')
            else:
                print('Изменения негативно повлияли на значение целевой функции')

            # nota bene: так как нумерация массивов начинается с нуля, номера векторов при выводе в консоль
            # мысленно увеличиваем на единицу для проверки

            break

