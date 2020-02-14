import sympy as sym
import typing as tp


def linearize_vector_valued_state_function(
        vector_valued_function: sym.MatrixBase,
        state_vector_symbols: tp.Tuple[sym.Symbol, ...]) \
        -> tp.Tuple[sym.MatrixBase, sym.MatrixBase]:
    """
    This function takes a vector involving state variables and computed the
    coefficients of a first order Taylor expansion around zero.

    Args:
        vector_valued_function: a column vector depending on the state
        state_vector_symbols: state vector as a tuple of symbols

    Returns: a tuple consisting of the constant and the linear coefficient in
             that order

    """

    # construct dictionary mapping state variables to zero:
    zero_state_dict = {state_variable: sym.numbers.Zero()
                       for state_variable
                       in state_vector_symbols}

    # generate constant coefficient
    constant_coefficient = vector_valued_function.subs(zero_state_dict)

    # generate linear coefficient
    linear_coefficient = sym.zeros(len(vector_valued_function),
                                   len(state_vector_symbols))
    for i in range(len(vector_valued_function)):
        for j, state_vector_symbol in enumerate(state_vector_symbols):
            matrix_element = \
                (vector_valued_function[i]
                 .diff(state_vector_symbol)
                 .subs(zero_state_dict))
            linear_coefficient[i, j] = matrix_element

    return constant_coefficient, linear_coefficient
