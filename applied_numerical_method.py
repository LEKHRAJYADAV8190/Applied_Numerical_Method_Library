
def placeholder_function():
    return 'This is a placeholder function in the Applied Numerical Method library.'

import sympy as sp

def newton_raphson(func_str, initial_guess, decimal_places=4, max_iterations=100):
    x = sp.symbols('x')
    try:
        func = sp.sympify(func_str)
        derivative = sp.diff(func, x)
        print(f"Derivate:{derivative}")
    except sp.SympifyError:
        print("Invalid function. Please ensure you're using the correct syntax.")
        return None

    tolerance = 10**(-decimal_places)

    x_n = initial_guess
    for n in range(max_iterations):
        f_x_n = func.subs(x, x_n).evalf()
        df_x_n = derivative.subs(x, x_n).evalf()

        print(f"Iteration {n + 1}:")
        print(f"   Current estimate of root: x_{n} = {x_n:.{decimal_places}f}")
        print(f"   Function value at this estimate: f(x_{n}) = {f_x_n:.{decimal_places}f}")
        print(f"   Derivative at this estimate: f'(x_{n}) = {df_x_n:.{decimal_places}f}")

        if df_x_n == 0:
            print("Zero derivative. No solution found.")
            return None

        x_n_plus_1 = x_n - f_x_n / df_x_n
        print(f"   Update: x_{n+1} = x_{n} - f(x_{n}) / f'(x_{n}) = {x_n_plus_1:.{decimal_places}f}")

        if abs(x_n_plus_1 - x_n) < tolerance:
            print(f"\nFound solution after {n + 1} iterations.")
            return x_n_plus_1

        x_n = x_n_plus_1
        print(f"   Updated estimate of root: x_{n+1} = {x_n:.{decimal_places}f}\n")

    print("Exceeded maximum iterations. No solution found.")
    return None

import sympy as sp

def bisection_method(func_str, a=-5, b=5, precision=0.0001, max_iterations=100):
    x = sp.symbols('x')
    try:
        func = sp.sympify(func_str)
    except sp.SympifyError:
        print("Invalid function. Please ensure you're using the correct syntax.")
        return None

    fa = func.subs(x, a).evalf()
    fb = func.subs(x, b).evalf()

    if fa * fb > 0:
        print("Invalid interval. The function values at the endpoints have the same sign.")
        return None

    iteration = 0

    while iteration < max_iterations:
        c = (a + b) / 2
        fc = func.subs(x, c).evalf()

        print(f"Iteration {iteration + 1}:")
        print(f"   Current interval: a = {a}, b = {b}")
        print(f"   Midpoint (c) and function value at c: c = {c}, f(c) = {fc}")

        if abs(fc) < precision:
            print(f"Found solution after {iteration + 1} iterations.")
            return c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        iteration += 1
        print(f"   Updated interval for next iteration: a = {a}, b = {b}\n")

    print("Exceeded maximum iterations. No solution found.")
    return None

import sympy as sp

def secant_method(func_str, x0=0.0, x1=1.0, decimal_places=4, max_iterations=100):
    x = sp.symbols('x')
    try:
        func = sp.sympify(func_str)
    except sp.SympifyError:
        print("Invalid function. Please ensure you're using the correct syntax.")
        return None

    tolerance = 10**(-decimal_places)

    for n in range(max_iterations):
        f_x0 = func.subs(x, x0).evalf()
        f_x1 = func.subs(x, x1).evalf()

        print(f"Iteration {n + 1}:")
        print(f"   Current estimates: x0 = {x0:.{decimal_places}f}, x1 = {x1:.{decimal_places}f}")
        print(f"   Function values: f(x0) = {f_x0}, f(x1) = {f_x1}")

        if f_x1 - f_x0 == 0:
            print("Zero difference in function values. No solution found.")
            return None

        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        print(f"   New estimate (x_new) calculation: x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))")
        print(f"   New estimate: x_new = {x_new:.{decimal_places}f}")

        if abs(x_new - x1) < tolerance:
            print(f"\nFound solution after {n + 1} iterations.")
            return x_new

        x0, x1 = x1, x_new
        print(f"   Updated estimates for next iteration: x0 = {x0:.{decimal_places}f}, x1 = {x1:.{decimal_places}f}\n")

    print("Exceeded maximum iterations. No solution found.")
    return None

import sympy as sp

def false_position_method(func_str, a=0, b=1, decimal_places=4, max_iterations=100):
    x = sp.symbols('x')
    try:
        func = sp.sympify(func_str)
        print(f"Function: {func}")
    except sp.SympifyError:
        print("Invalid function. Please ensure you're using the correct syntax.")
        return None

    tolerance = 10**(-decimal_places)
    fa = func.subs(x, a).evalf()
    fb = func.subs(x, b).evalf()

    print(f"Initial interval: [a={a}, b={b}]")
    print(f"Function values at the initial points: f(a)={fa}, f(b)={fb}")

    if fa * fb > 0:
        print("Function values at the initial points must have opposite signs.")
        return None

    for n in range(max_iterations):
        c = b - fb * (b - a) / (fb - fa)
        fc = func.subs(x, c).evalf()

        print(f"Iteration {n + 1}:")
        print(f"   Current interval: a = {a:.{decimal_places}f}, b = {b:.{decimal_places}f}")
        print(f"   Function values: f(a) = {fa:.{decimal_places}f}, f(b) = {fb:.{decimal_places}f}")
        print(f"   False Position Formula: c = b - f(b) * (b - a) / (f(b) - f(a))")
        print(f"   New estimate: c = {c:.{decimal_places}f}, f(c) = {fc:.{decimal_places}f}")

        if abs(fc) < tolerance:
            print(f"\nFound solution after {n + 1} iterations.")
            return c

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

        print(f"   Updated interval for next iteration: a = {a:.{decimal_places}f}, b = {b:.{decimal_places}f}")

    print("Exceeded maximum iterations. No solution found.")
    return None







    

import sympy

from sympy import symbols, simplify, N

def create_difference_table(y_data):
    table = [y_data]
    while len(table[-1]) > 1:
        table.append([j - i for i, j in zip(table[-1], table[-1][1:])])
    return table

def format_difference_table1(x_data, table):
    formatted_table = "x\ty\t" + "\t".join(["Δ^{}y".format(i) for i in range(1, len(table))]) + "\n"
    for i in range(len(x_data)):
        formatted_row = f"{x_data[i]}\t" + "\t".join(f"{round(table[j][i], 5) if i < len(table[j]) else ''}" for j in range(len(table)))
        formatted_table += formatted_row + "\n"
    return formatted_table

def round_small_values(expr, tolerance=1e-10):
    # Function to round off small values in an expression
    return expr.xreplace({n : 0 if abs(n) < tolerance else round(n, 5) for n in expr.atoms(sympy.Number)})

def newton_forward_interpolation(x_data, y_data, x_value):
    x = symbols('x')
    n = len(y_data)
    h = x_data[1] - x_data[0]  # Assuming equal spacing
    s = (x - x_data[0]) / h

    # Set up the difference table
    difference_table = create_difference_table(y_data)

    # Apply Newton's Forward Formula
    P_x = y_data[0]
    s_product = 1
    for i in range(1, n):
        s_product *= (s - i + 1) / i
        delta_term = s_product * difference_table[i][0]
        P_x += delta_term

    # Round small values and simplify
    P_x_rounded = round_small_values(P_x)
    P_x_simplified = N(simplify(P_x_rounded).subs(x, x_value), 5)

    # Print the results
    print("Newton's Forward Interpolation Formula:")
    print("P(x) = f(a) + (s/1!) Δf(a) + (s(s-1)/2!) Δ²f(a) + ... + (s(s-1)...(s-n+1)/n!) Δⁿf(a)")
    print("Where s = (x - a) / h\n")
    print("Step Size (h):", h)
    print(f"Calculating 's' for x = {x_value}: s = {(x_value - x_data[0]) / h}\n")
    print("Difference Table:")
    print(format_difference_table1(x_data, difference_table))
    print(f"P(x) = {P_x_rounded}")
    print(f"Simplified P({x_value}): {P_x_simplified}")

    return P_x_simplified

from sympy import symbols, simplify, N

def create_backward_difference_table(y_data):
    n = len(y_data)
    table = [y_data.copy()]
    for i in range(1, n):
        row = []
        for j in range(n-i, 0, -1):
            diff = table[-1][j] - table[-1][j-1]
            row.insert(0, diff)
        table.append(row)
    return table

def format_difference_table2(x_data, table):
    formatted_table = "x\ty\t" + "\t".join(["Δ^{}y".format(i) for i in range(1, len(table))]) + "\n"
    for i in range(len(x_data)):
        formatted_row = f"{x_data[i]}\t" + "\t".join(f"{round(table[j][i-j], 5) if i-j >= 0 else ''}" for j in range(len(table)))
        formatted_table += formatted_row + "\n"
    return formatted_table

def round_small_values(expr, tolerance=1e-10):
    return expr.xreplace({n : 0 if abs(n) < tolerance else round(n, 5) for n in expr.atoms(sympy.Number)})

def newton_backward_interpolation(x_data, y_data, z_value):
    x = symbols('x')
    n = len(y_data)
    h = x_data[1] - x_data[0]
    u = (x - x_data[-1]) / h

    # Set up the backward difference table
    difference_table = create_backward_difference_table(y_data)

    # Apply Newton's Backward Formula
    P_x = y_data[-1]
    u_product = 1
    for i in range(1, n):
        u_product *= (u + i - 1) / i
        delta_term = u_product * difference_table[i][-1]
        P_x += delta_term

    # Round small values and simplify
    P_x_rounded = round_small_values(P_x)
    P_x_simplified = N(simplify(P_x_rounded).subs(x, z_value), 5)

    # Print the results
    print("Newton's Backward Interpolation Formula:")
    print("P(x) = f(b) + (u/1!) Δf(b) + (u(u+1)/2!) Δ²f(b) + ... + (u(u+1)...(u+n-1)/n!) Δⁿf(b)")
    print("Where u = (x - b) / h and b is the last data point\n")
    print("Step Size (h):", h)
    print(f"Calculating 'u' for x = {z_value}: u = {(z_value - x_data[-1]) / h}\n")
    print("Backward Difference Table:")
    print(format_difference_table2(x_data, difference_table))
    print(f"\nP(x) = {P_x_rounded}")
    print(f"Simplified P({z_value}): {P_x_simplified}\n")

    return P_x_simplified

import numpy as np
from sympy import symbols, simplify, N

# Function to create the divided difference table based on the formula given in the image
def create_divided_difference_table(x_data, y_data):
    n = len(y_data)
    divided_diff_table = np.zeros((n, n))
    divided_diff_table[:, 0] = y_data
    
    for j in range(1, n):
        for i in range(n - j):
            divided_diff_table[i][j] = (divided_diff_table[i+1][j-1] - divided_diff_table[i][j-1]) / (x_data[i+j] - x_data[i])
    
    return divided_diff_table

# Function to format the divided difference table
def format_divided_difference_table(x_data, divided_diff_table):
    formatted_table = "Divided Difference Table:\n"
    formatted_table += "---------------------------------------------------\n"
    headers = ["x", "f(x)"] + [f"Δ^{i}y" for i in range(1, len(x_data))]
    header_row = "{:>8}" * len(headers) + "\n"
    formatted_table += header_row.format(*headers)
    
    for i in range(len(x_data)):
        row_data = [x_data[i]] + list(divided_diff_table[i, :len(x_data) - i])
        row_format = "{:>8}" + "{:>12}" * (len(row_data) - 1) + "\n"
        formatted_table += row_format.format(*row_data)
    
    return formatted_table.strip()

# Function to display the general form of Newton's Divided Difference Formula
def display_newton_formula():
    formula = "Applying Newton’s divided difference formula:\n"
    formula += "f(x) = f(x0) + (x - x0)[x0, x1] + (x - x0)(x - x1)[x0, x1, x2] + ..."
    return formula

# Function to construct the Newton divided difference polynomial
def construct_newton_polynomial(x_data, divided_diff_table):
    x_sym = symbols('x')
    polynomial = divided_diff_table[0, 0]
    for i in range(1, len(x_data)):
        term = divided_diff_table[0, i]
        for j in range(i):
            term *= (x_sym - x_data[j])
        polynomial += term
    return simplify(polynomial)

# Function to evaluate the Newton polynomial at a given x value
def evaluate_newton_polynomial(poly, x_value):
    x_sym = symbols('x')
    return N(poly.subs(x_sym, x_value), 5)

# Main function to process the data and output the results
def newton_divided_difference(x_data, y_data, x_value):
    # Calculate and format the divided difference table
    divided_diff_table = create_divided_difference_table(x_data, y_data)
    formatted_table = format_divided_difference_table(x_data, divided_diff_table)
    
    # Display the divided difference table
    print(formatted_table)
    
    # Display the Newton's divided difference formula
    print("\n" + display_newton_formula())
    
    # Construct and simplify the Newton polynomial
    newton_poly = construct_newton_polynomial(x_data, divided_diff_table)
    
    # Display the Newton polynomial
    print("\nNewton's Divided Difference Polynomial P(x):")
    print(newton_poly)
    
    # Evaluate the polynomial at the given x value
    evaluated_value = evaluate_newton_polynomial(newton_poly, x_value)
    
    # Display the evaluated value
    print(f"\nP({x_value}) = {evaluated_value}")

    return evaluated_value

import sympy as sp

def lagrange_interpolation(x_vals, y_vals, value):
    x = sp.symbols('x')
    L = 0
    step_expressions = []

    # Calculating the Lagrange Polynomial and storing step expressions
    for i in range(len(y_vals)):
        term = y_vals[i]
        term_expression = f'y{i} term: ' + ('+ ' if i > 0 else '') + f'y{i} * '
        for j in range(len(x_vals)):
            if i != j:
                factor = (x - x_vals[j]) / (x_vals[i] - x_vals[j])
                term *= factor
                term_expression += f'((x - {x_vals[j]}) / ({x_vals[i] - x_vals[j]})) * '
        term_expression = term_expression.rstrip(' * ')  # Remove trailing asterisk
        L += term
        step_expressions.append(term_expression)

    simplified_L = sp.simplify(L)
    L_at_value = simplified_L.subs(x, value)

    # Displaying the results

    print("\nIntermediate Steps:")
    for step in step_expressions:
        print(step)
    
    
    print("Simplified Polynomial:")
    print(simplified_L)

    print("\nEvaluated Value at x =", value, ":")
    print(L_at_value)



import sympy as sp
from sympy import solve, Eq

def cubic_spline(x_data, y_data, z):
    x = sp.symbols('x')
    M = sp.symbols(f'M0:{len(x_data)}')

    equations = [
        Eq(M[0], 0),  # Natural boundary condition at the start
        Eq(M[len(x_data) - 1], 0)  # Natural boundary condition at the end
    ]

    # Assume that the x values are equally spaced
    h = x_data[1] - x_data[0]  # Calculate h based on the first two x values
    
    # Generating the middle equations
    for i in range(1, len(x_data) - 1):
        eq = Eq(M[i - 1] + 4 * M[i] + M[i + 1], 6 / h**2 * (y_data[i - 1] - 2 * y_data[i] + y_data[i + 1]))
        equations.append(eq)

    # Solve the system for the M values
    solutions = solve(equations, M)

    # Define the piecewise spline function
    spline_pieces = [
        1/(6 * h) * (y_data[i] * (x_data[i + 1] - x)**3 + y_data[i + 1] * (x - x_data[i])**3)
        + (y_data[i] / h - solutions[M[i]] * h / 6) * (x_data[i + 1] - x)
        + (y_data[i + 1] / h - solutions[M[i + 1]] * h / 6) * (x - x_data[i])
        for i in range(len(x_data) - 1)
    ]

    # Print the cubic spline equations
    print("Cubic spline equations:")
    for i, piece in enumerate(spline_pieces):
        equation_str = str(piece)
        equation_str = equation_str.replace('**', '^')
        print(f"For x in [{x_data[i]}, {x_data[i + 1]}]:")
        print(equation_str)

    # Evaluate the spline at the given point z
    for i in range(len(x_data) - 1):
        if x_data[i] <= z <= x_data[i + 1]:
            spline_value = spline_pieces[i].subs(x, z).evalf()
            print(f"Spline value at z = {z} is approximately {spline_value:.4f}")
            return spline_value

    raise ValueError("The evaluation point z is outside the range of x_data.")






















