import numpy as np
from sympy import symbols, diff


# for 1d array of functions.
def substitution(a, x_values):
    n = len(a)
    x1, x2 = symbols('x1,x2')
    result = []
    for i in range(n):
        expr = eval(a[i])
        expr = expr.subs(x1, x_values[0])
        expr = expr.subs(x2, x_values[1])
        result.append(expr)
    return result


# for a matrix
def substitution_jacobian(a, x_values):
    n = len(a)
    x1, x2 = symbols('x1,x2')
    result = np.zeros([n, n])
    for i in range(len(a)):
        for j in range(len(a[i])):

            expr = eval(a[i][j])
            if type(expr) == int:
                result[i][j] = expr
            else:
                expr = expr.subs(x1, x_values[0])
                expr = expr.subs(x2, x_values[1])
                result[i][j] = expr
    return result


def broyden():

    # initial guess
    x = np.array([1, 2])
    # functions
    f = ['x1 + 2 * x2 - 2', "x1**2 + 4 * x2**2 - 4" ]

    # initial setup for Broyden's method
    f_line = substitution(f, x)
    f_line = np.array(f_line)

    j = jacobian(f)
    b_line = substitution_jacobian(j, x)

    tolerance = 1e-4
    current_err = 100 * tolerance
    k = 1

    print(" {:<30} {:20} {:<15} {:<10}".format("B", "x", "f(x)", "K"))
    while(current_err > tolerance):

        x_old = x
        f_old = f_line
        b_old = np.array(b_line)

        # 1. get sk
        sk = gauss(b_old, np.negative(f_old))

        # 2.  get xk : old x + sx
        x = x_old + sk

        # 3. update fx
        f_line = substitution(f, x)

        # 4. get yk
        y = np.array(f_line) - np.array(f_old)

        # 5.  get Bk -- correct default orientation
        y_transpose = np.transpose(np.matrix(y))
        sk_transpose = np.transpose(np.matrix(sk))

        b_between = np.outer((y_transpose - np.dot(b_old, sk_transpose)),sk) / (np.dot(sk, sk_transpose))
        b_line = b_old + b_between

        current_err = abs(x[1] - x_old[1])

        #format the arrays
        x_formated = np.around(x,4)
        f_line_formated = np.around( np.array(f_line).astype(np.double), 4)
        b_formated = np.matrix.round(b_line.astype(np.double), 4)

        print_chart(k, b_formated, x_formated, f_line_formated)
        k = k + 1


def print_chart(iterations, B, x, fx):

    str_b = "{:10}".format(str(B))
    str_fx ="{:10}".format(str(fx))
    str_x = "{:6}".format(str(x))
    print("{:<40} {:<25} {:<20} {:<10} ".format(str_b, str_x, str_fx, iterations))


def jacobian(functions):

    n = len(functions)
    x1, x2 = symbols('x1 x2', real=True)
    result = [[], []]
    counter = 0
    for i in range(n):
        expr = eval(functions[i])
        x1_diff = diff(expr,x1)
        x2_diff = diff(expr, x2)
        result[counter].append([str(x1_diff), str(x2_diff)])
        counter += 1
    result = np.array(result)
    result = result.reshape(2,2)
    return result


# used to solve system of equations
def backward_substitution(matrix, x, b):
    n = len(matrix) - 1
    for j in range(n, -1, -1):
        if matrix[j, j] == 0:
            break
        x[j] = b[j] / matrix[j][j]
        for i in range(j):
            b[i] = b[i] - (matrix[i, j] * x[j])
    return x


def gauss(a, b):
    n = len(a)
    multiple = np.identity(n)
    x = np.zeros(n) # x
    copy_arr = np.copy(a)
    ratio = 0
    for k in range(n):
        if copy_arr[k][k] == 0.0:
            break
        for i in range(k + 1, n):
            multiple[i][k] = (copy_arr[i][k] / copy_arr[k][k])
            ratio = copy_arr[i][k] / copy_arr[k][k]
            for j in range(n):
                copy_arr[i][j] = copy_arr[i][j] - (ratio * copy_arr[k, j])
            b[i] = b[i] - (ratio * b[k])
    res = backward_substitution(copy_arr, x, b)
    return res

broyden()





