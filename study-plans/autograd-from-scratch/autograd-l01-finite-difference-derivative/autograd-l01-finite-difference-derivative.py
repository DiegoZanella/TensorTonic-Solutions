import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    poly_value_at_x = sum(
        [
            coefficients[k] * x**k
            for k in range(len(coefficients))
        ]
    )

    poly_value_at_step = sum(
        [
            coefficients[k] * (x+h)**k
            for k in range(len(coefficients))
        ]
    )

    forward_slope = (poly_value_at_step - poly_value_at_x) / h
    return poly_value_at_x, poly_value_at_step, forward_slope
