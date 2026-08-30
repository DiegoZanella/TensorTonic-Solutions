import numpy as np

def scalar_expression_partials(a, b, c, h):
    """
    Returns: the expression value and its three numerical partial derivatives
    """
    scalar_at_x = a*b + c
    partial_a = ( (a+h)*b + c -scalar_at_x) / h 
    partial_b = ( 
        a*(b+h) + c -scalar_at_x
    ) / h 
    partial_c = ( 
        a*b + (c+h) - scalar_at_x
    ) / h 
    return scalar_at_x, partial_a, partial_b, partial_c
