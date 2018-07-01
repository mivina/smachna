import numpy as np
import math
import matplotlib.pyplot as plt


def yd(x, y):
    """ y''=sin x
    k' = sin x
    y' = k
    y0' = sin x
    y1' = y0
    """
    return np.array([math.sin(x),
                     y[0]
                     ])


def rk3(x0, x_end, step, y0_list, f=yd):
    """Runge–Kutta 3 method
    :param x0: begin
    :param x_end: end
    :param step: step
    :param y0_list: list of y(x0)
    :param f: function (x, y_list) -> y'(x)_list
    :return: matrix with columns: x, y_1, y_2, ..., y_n
    """
    assert x_end > x0
    n = round((x_end - x0) / step)
    n = 1 if n < 1 else n
    h = (x_end - x0) / n
    yk = np.array(y0_list, dtype='f')
    xk = x0
    ans = np.ndarray(shape=(1, 1 + len(yk)))
    ans[0] = np.insert(yk, 0, xk)
    for i in range(n + 1):
        k1 = h * f(xk, yk)
        k2 = h * f(xk + h / 3, yk + k1 / 3)
        k3 = h * f(xk + 2 * h / 3, yk + 2 * k2 / 3)
        yk = yk + (k1 + 3 * k3) / 4
        xk = x0 + (i + 1) * h
        ans = np.append(ans, np.insert(yk, 0, xk).reshape(1, 1 + len(yk)), axis=0)
    return ans


def rk4(x0, x_end, step, y0_list, f=yd):
    """Runge–Kutta 4 method
    :param x0: begin
    :param x_end: end
    :param step: step
    :param y0_list: list of y(x0)
    :param f: function (x, y_list) -> y'(x)_list
    :return: matrix with columns: x, y_1, y_2, ..., y_n
    """
    n = round((x_end - x0) / step)
    n = 1 if n < 1 else n
    h = (x_end - x0) / n
    yk = np.array(y0_list, dtype='f')
    xk = x0
    ans = np.ndarray(shape=(1, 1 + len(yk)))
    ans[0] = np.insert(yk, 0, xk)
    for i in range(n):
        k1 = h * f(xk, yk)
        k2 = h * f(xk + h / 2, yk + k1 / 2)
        k3 = h * f(xk + h / 2, yk + k2 / 2)
        k4 = h * f(xk + h, yk + k3)
        yk = yk + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        xk = x0 + (i + 1) * h
        ans = np.append(ans, np.insert(yk, 0, xk).reshape(1, 1 + len(yk)), axis=0)
    return ans


def adams_bashford_5(x0, x_end, y0_list, step, f=yd, one_step_method=rk4):
    """Adams-Bashford 5 method
    :param x0: begin
    :param x1: end
    :param step: step
    :param y0_list: list of y(x0)
    :param f: function (x, y_list) -> y'(x)_list
    :param one_step_method: method to calculate first 5 points
    :return: matrix with columns: x, y_1, y_2, ..., y_n
    """
    assert x_end > x0
    n = round((x_end - x0) / step)
    n = 1 if n < 1 else n
    h = (x_end - x0) / nh
    xy_first = one_step_method(x0, x0 + 4 * h, step=h, y0_list=y0_list, f=f)
    yk = np.array(xy_first[4, 1:], dtype='f')
    fk = np.zeros_like(xy_first[:, 1:])
    for i in range(len(fk)):
        fk[i] = f(xy_first[i, 0], xy_first[i, 1:])
    ans = np.ndarray(shape=(5, 1 + len(yk)))
    if n < 6:
        print('Incorrect method')
        return xy_first
    ans[:5] = xy_first
    for i in range(4, n):
        yk = yk + h / 720 * (1901 * fk[-1] - 2774 * fk[-2] + 2616 * fk[-3] - 1274 * fk[-4] + 251 * fk[-5])
        xk = x0 + (i + 1) * h
        ans = np.append(ans, np.insert(yk, 0, xk).reshape(1, 1 + len(yk)), axis=0)
        fk = np.append(fk, f(xk, yk).reshape(1, len(yk)), axis=0)
    return ans


def thomas(upper, central, low, b):
    """Solve three-diagonal SLAE by Thomas method
    :param upper: array, upper diagonal
    :param central: array, main diagonal
    :param low: array, low diagonal
    :param b: array, right part of the SLAE
    :return: numpy.ndarray, solution
    """
    neq = len(central)
    ans = [0] * neq
    for i in range(1, neq):
        central[i] -= (upper[i - 1] * low[i - 1]) / central[i - 1]
        b[i] -= (low[i - 1] * b[i - 1]) / central[i - 1]
    ans[neq - 1] = (b[neq - 1] / central[neq - 1])
    for j in range(neq - 2, -1, -1):
        ans[j] = (b[j] - upper[j] * ans[j + 1]) / central[j]
    return np.array(ans)


def ydRIB(x, y):
    u = 0.25
    k = [1.211, 22.188, 12.618, 0.044]
    ans = np.array([1 / u * (-k[0] * y[0] + k[3] * y[1]),
                     1 / u * (k[0] * y[0] - k[1] * y[1] + k[2] * y[2] - k[3] * y[1]),
                     1 / u * (k[1] * y[1] - k[2] * y[2])
                     ])
    return ans


def ydRIC(x, y):
    t = 4
    ci = [0.265, 0, 0]
    k = [1.211, 22.188, 12.618, 0.044]
    ans = np.array([1 / t * (ci[0] - y[0]) - k[0] * y[0] + k[3] * y[1],
                     1 / t * (ci[1] - y[1]) + k[0] * y[0] - k[1] * y[1] + k[2] * y[2] - k[3] * y[1],
                     1 / t * (ci[2] - y[2]) + k[1] * y[1] - k[2] * y[2]
                     ])
    return ans


if __name__ == '__main__':
    answer = adams_bashford_5(x0=0, x_end=3, step=0.001, y0_list=[0.265, 0, 0], f=ydRIB)
    xp = answer[:, 0]
    y1 = answer[:, 1]
    y2 = answer[:, 2]
    y3 = answer[:, 3]
    plt.subplot(2, 1, 1)
    plt.plot(xp, y1, label='Ca')
    plt.plot(xp, y2, label='Cb')
    plt.plot(xp, y3, label='Cc')
    plt.grid(True)
    plt.legend()
    plt.title('РИВ')

    answer = adams_bashford_5(x0=0, x_end=3, step=0.001, y0_list=[0.265, 0, 0], f=ydRIC)
    xp = answer[:, 0]
    y1 = answer[:, 1]
    y2 = answer[:, 2]
    y3 = answer[:, 3]
    plt.subplot(2, 1, 2)
    plt.plot(xp, y1, label='Ca')
    plt.plot(xp, y2, label='Cb')
    plt.plot(xp, y3, label='Cc')
    plt.grid(True)
    plt.legend()
    plt.title('РИС')
    # ans = rk3(0, 3.14, 0.01, [-1, 0])
    # x = np.linspace(0, 1, 3.14/0.01+2)
    # y1 = -1 * np.sin(x)
    # plt.plot(x, ans[:, 2])
    # plt.plot(x, ans[:, 1])
    #
    # print(ans)
    # eps = np.abs(y1 - ans[:, 2])
    # for i in eps:
    #     print(i)
    '''plt.plot(
    xp, yp, label='y\'')
    plt.plot(xp, y, label='y')
    plt.ylabel('Y')
    plt.title('Solution')
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(xp, yp + np.cos(xp), label="y'", color='orange')
    plt.plot(xp, y + np.sin(xp), label='y', color='r')
    plt.xlabel('X')
    plt.title('Absolute error')
    plt.grid(True)'''
    plt.tight_layout()
    plt.show()
# проверка связи
все працює
