"""
A collection of simple math operations
"""

def simple_add(a,b):
    """
    Purpose: 
        Add two numbers together.
    Parameters:
        a : float or int - First number.
        b : float or int - Second number.
    Returns:
        float or int - The sum of a and b, i.e. the result of a + b.
    """
    return a+b

def simple_sub(a,b):
    """
    Purpose:
        Subtract one number from another.
    Parameters:
        a : float or int - First number.
        b : float or int - Second number.
    Returns:
        float or int - b subtracted from a, i.e. the result of a - b.
    """
    return a-b

def simple_mult(a,b):
    """
    Purpose:
        Multiply two numbers.
    Parameters:
        a : float or int - First number.
        b : float or int - Second number.
    Returns:
        float or int - The product of a and b, i.e. the result of a * b.
    """
    return a*b

def simple_div(a,b):
    """
    Purpose:
        Divide one number by another.
    Parameters:
        a : float or int - Numerator.
        b : float or int - Denominator (must fulfill b != 0).
    Returns:
        float: a divided by b, i.e. the result of a / b.
    Raises:
        ZeroDivisionError : If b == 0.   
    """
    return a/b

def poly_first(x, a0, a1):
    """
    Purpose:
        Evaluate a first degree polynomial of the form f(x) = a0 + a1 * x
    Parameters:
        x : float or int - The input variable.
        a0 : float or int - The constant term.
        a1 : float or int - The coefficient of x.
    Returns: 
        float or int - The calculated value of the polynomial, i.e. the result of a0 + a1 * x.
    """
    return a0 + a1*x

def poly_second(x, a0, a1, a2):
    """
    Purpose:
        Evaluate a second degree polynomial of the form f(x) = a0 + a1 * x + a2 * x ** 2
    Parameters:
        x : float or int - The input variable.
        a0 : float or int - The constant term.
        a1 : float or int - The coefficient of x.
        a2 : float or int - The coefficient of x**2
    Returns:
        float or int - The calculated value of the polynomial, i.e. the result of a0 + a1 * x + a2 * x ** 2
    """
    return poly_first(x, a0, a1) + a2*(x**2)

# Feel free to expand this list with more interesting mathematical operations...
# .....
