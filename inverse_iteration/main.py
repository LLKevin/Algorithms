import numpy as np
from numpy.linalg import norm

def gauss(a):
    n = len(a)
    multiple = np.identity(n)
    m = np.array([[1,0,0], [0, 1, 0], [0, 0, 1]])
    result = np.zeros((n, n))
    ratio = 0

    for k in range(n):
        if a[k][k] == 0.0:
            break
        for i in range(k + 1, n):
            multiple[i][k] = (a[i][k] / a[k][k])
            ratio = a[i][k] / a[k][k]
            for j in range(n):
                a[i][j] = a[i][j] - (ratio * a[k, j])

    for i in range(n):
        res = forward_substition(multiple, np.zeros(n), m[i])
        result[i] = (backward_substition(a, np.zeros(n), res))
        print(res)
    return result


def backward_substition(matrix, x, b):
    n = len(matrix) - 1
    for j in range(n, -1, -1):
        if matrix[j, j] == 0:
            break
        x[j] = b[j] / matrix[j][j]
        for i in range(j):
            b[i] = b[i] - (matrix[i, j] * x[j])
    return x


def forward_substition(matrix, x, b):
    n = len(matrix)
    for i in range(n):
        temp = b[i]
        for j in range(n):
            temp = temp - (matrix[i][j] * x[j])
        x[i] = (temp / matrix[i][i])
    return x


def inverse_interation(a):
    print("{:<8} {:<15} {:<10}".format("K", "xK", "Ration/ Lambda"))
    tolerance = 0.001
    change = 1
    ratio = 100
    counter = 0
    x = np.array([0, 1,1])
    inverse = gauss(a)

    while change > tolerance:
        xold = x
        x = np.dot(inverse, x)
        norm_x0 = norm(x, np.inf)
        x = x / norm_x0
        print_chart(counter, x, norm_x0)
        counter += 1
        change = abs(norm_x0 - ratio) / ratio * 100
        ratio = norm_x0


def print_chart(count, vector, ratio):
    # print("{:<8} {:<15} {:<10}")
    x = "{:9}".format(str(vector))
    print("{:<8} {:<15} {:<10} ".format(count, x, ratio))


a_test = np.array([[1,-1,0],[0,-4,2],[0,0,-2]], dtype=float)
inverse_interation(a_test)

