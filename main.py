import numpy as np 
import matplotlib.pyplot as plt


def odchylenie_standardowe(n, p, blad_apro):
    return np.sqrt(1 / (n - p) * np.sum(blad_apro ** 2))

def wariancja_resztowa():
    return None


def f_liniowy(xi, p0, p1):
    return p0 + p1 * xi

def aproksymacja_liniowa(x, y, k):
    n = x.shape[0]

    X = np.zeros((2,2), dtype=float)
    Y = np.zeros(2, dtype=float)

    for i in range(n):
        X[0][0]+=1
        X[0][1]+=x[i]
        X[1][0] = X[0][1]
        X[1][1]+=x[i]**2

        Y[0] += y[i]
        Y[1] += x[i] * y[i]

    P = np.linalg.solve(X, Y)
    print(f'wspolczynniki P: \n{P}')

    P_bazowa = np.polyfit(x, y, 1)
    P_bazowa[0], P_bazowa[1] = P_bazowa[1], P_bazowa[0]
    roznica = P - P_bazowa
    print("roznica w wynikach miedzy rezultatem a wartosciami oczekiwanymi:")
    print(np.array2string(roznica, formatter={'float_kind': lambda x: "%.6f" % x}))
    print()


    blad_apro_square_sum = 0
    for i in range(n):
        blad_apro_square_sum += ( y[i] - f_liniowy(x[i], P[0], P[1]) ) ** 2
    wariacja_resztowa = 1 / (n - 2) * blad_apro_square_sum
    print(f'wariacja resztowa: {wariacja_resztowa:.6f}')
    print(f'odchylene standardowe: {np.sqrt(wariacja_resztowa):.6f}')

    y_avg = np.average(y)
    r2 = 1 - (blad_apro_square_sum / np.sum( (y - np.full(n, y_avg)) ** 2))
    print(f'wzpolczynnik determinacji: {r2:.6f}')


    x_apr = np.linspace(x[0], x[-1], k)
    y_apr = np.zeros(k, dtype=float)
    for i in range(k):
        y_apr[i] = f_liniowy(x_apr[i], P[0], P[1])

    return x_apr, y_apr

print(f'\n\n{10*'*'} aproksymacja liniowa {10*'*'}\n\n')

x = np.array([1.15, 1.71, 2.96, 4.24, 5.25, 5.84, 6.83, 8.06, 
              9.14, 10.12, 10.91, 11.85, 13.14, 14.05, 15.04], dtype=float)
y = np.array([1.6, 25.08, 14.25, 2.29, 45.3, 46.85, 74.99, 86.8,
               97.44, 154.48, 153.87, 188.07, 240.71, 243.66, 255.87], dtype=float)

x_apr, y_apr = aproksymacja_liniowa(x, y, 500)


plt.figure()
plt.scatter(x, y,  marker='o', color='green', label='punkty')
plt.plot(x_apr, y_apr, color='red', label='funkcja aproksymacyjna')
plt.title("Aproksymacja liniowa")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

print()
print()





def f_kwadratowy(xi, p0, p1, p2):
    return p0 + (p1 * xi) + p2 * (xi**2)

def aproksymacja_kwadratowa(x, y, k):
    n = x.shape[0]

    X = np.zeros((3,3), dtype=float)
    Y = np.zeros(3, dtype=float)

    for i in range(n):
        X[0][0]+=1

        X[0][1]+=x[i]
        X[1][0] = X[0][1]

        X[0][2]+=x[i]**2
        X[1][1] = X[2][0] = X[0][2]

        X[1][2] += x[i]**3
        X[2][1] = X[1][2]

        X[2][2] += x[i]**4

        Y[0] += y[i]
        Y[1] += x[i] * y[i]
        Y[2] += (x[i]**2) * y[i]

    P = np.linalg.solve(X, Y)
    print(f'wspolczynniki P: \n{P}')

    P_bazowa = np.polyfit(x, y, 2)
    P_bazowa[0], P_bazowa[1], P_bazowa[2] = P_bazowa[2], P_bazowa[1], P_bazowa[0]
    roznica = P - P_bazowa
    print("roznica w wynikach miedzy rezultatem a wartosciami oczekiwanymi:")
    print(np.array2string(roznica, formatter={'float_kind': lambda x: "%.6f" % x}))
    print()

    blad_apro_square_sum = 0
    for i in range(n):
        blad_apro_square_sum += ( y[i] - f_kwadratowy(x[i], P[0], P[1], P[2]) ) ** 2
    wariacja_resztowa = 1 / (n - 3) * blad_apro_square_sum
    print(f'wariacja resztowa: {wariacja_resztowa:.6f}')
    print(f'odchylene standardowe: {np.sqrt(wariacja_resztowa):.6f}')

    x_apr = np.linspace(x[0], x[-1], k)
    y_apr = np.zeros(k, dtype=float)
    for i in range(k):
        y_apr[i] = f_kwadratowy(x_apr[i], P[0], P[1], P[2])

    return x_apr, y_apr

print(f'\n\n{10*'*'} aproksymacja kwadratowa {10*'*'}\n\n')

x = np.array([1.15, 1.71, 2.96, 4.24, 5.25, 5.84, 6.83, 8.06, 
              9.14, 10.12, 10.91, 11.85, 13.14, 14.05, 15.04], dtype=float)
y = np.array([1.6, 25.08, 14.25, 2.29, 45.3, 46.85, 74.99, 86.8,
               97.44, 154.48, 153.87, 188.07, 240.71, 243.66, 255.87], dtype=float)

x_apr, y_apr = aproksymacja_kwadratowa(x, y, 500)

plt.figure()
plt.scatter(x, y,  marker='o', color='green', label='punkty')
plt.plot(x_apr, y_apr, color='red', label='funkcja aproksymacyjna')
plt.title("Aproksymacja kwadratowa")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

print()
print()
