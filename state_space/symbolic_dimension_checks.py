import collections
import numbers
import typing as tp

import numpy as np
import sympy as sym

SympyMatrixCandidate = tp.Union[sym.Expr, sym.Array, sym.MatrixBase]


def is_sympy_scalar(scalar: SympyMatrixCandidate,
                    attempt_fix: bool = True) \
        -> tp.Tuple[bool, tp.Optional[sym.Expr]]:
    if not isinstance(scalar, sym.Expr):
        if attempt_fix and isinstance(scalar, np.ndarray) and scalar.size == 1:
            scalar = sym.Expr(scalar.item())
        elif isinstance(scalar, sym.MatrixBase):
            if scalar.cols == 1 and scalar.rows == 1 and attempt_fix:
                scalar = scalar[0, 0]
            else:
                return False, None
        else:
            return False, None

    return True, scalar


def check_sympy_scalar(scalar: SympyMatrixCandidate,
                       attempt_fix: bool = True,
                       label: str = "sympy expression") -> sym.Expr:
    if not isinstance(scalar, sym.Expr):
        if attempt_fix and isinstance(scalar, np.ndarray):
            if not scalar.size == 1:
                raise ValueError(f"{label} must be a sympy scalar but is a "
                                 f"{scalar.shape} numpy array")
            else:
                scalar = sym.sympify(scalar.item())
        elif attempt_fix and isinstance(scalar, numbers.Number):
            scalar = sym.sympify(scalar)
        else:
            raise TypeError(f"{label} must be of type sympy.Expr")

    if isinstance(scalar, sym.Matrix):
        if scalar.cols == 1 and scalar.rows == 1 and attempt_fix:
            scalar = scalar[0, 0]
        else:
            raise TypeError(f"{label} must be of type sympy.Expr but is a "
                            f"{scalar.rows}x{scalar.cols} sympy.Matrix")

    return scalar


def _check_conditions_for_conversion_to_matrix(matrix: SympyMatrixCandidate) \
        -> bool:
    if isinstance(matrix, (sym.Expr, collections.Sequence)):
        return True
    elif isinstance(matrix, (np.ndarray,
                             sym.Array)) \
            and matrix.ndim <= 2:
        return True
    else:
        return False


def check_sympy_matrix(matrix: SympyMatrixCandidate,
                       attempt_fix: bool = True,
                       label: str = "sympy matrix") -> sym.MatrixBase:
    if not isinstance(matrix, sym.MatrixBase):
        if attempt_fix and _check_conditions_for_conversion_to_matrix(matrix):
            matrix = sym.Matrix(matrix)
        else:
            raise TypeError(f"{label} must be of type sympy.MatrixBase")

    return matrix


def check_column_vector_sympy_expression_and_get_dimension(
        matrix: SympyMatrixCandidate,
        attempt_fix: bool = True,
        label: str = "sympy matrix") -> tp.Tuple[sym.MatrixBase, int]:
    matrix = check_sympy_matrix(matrix=matrix,
                                attempt_fix=attempt_fix,
                                label=label)

    if matrix.cols > 1:
        raise ValueError(f"{label} must be a column vector but is a "
                         f"{matrix.rows}x{matrix.cols} matrix")
    else:
        return matrix, matrix.rows


def check_row_vector_sympy_expression_and_get_dimension(
        matrix: SympyMatrixCandidate,
        attempt_fix: bool = True,
        label: str = "sympy matrix") -> tp.Tuple[sym.MatrixBase, int]:
    matrix = check_sympy_matrix(matrix=matrix,
                                attempt_fix=attempt_fix,
                                label=label)

    if matrix.rows > 1:
        raise ValueError(f"{label} must be a column vector but is a "
                         f"{matrix.rows}x{matrix.cols} matrix")
    else:
        return matrix, matrix.cols


def check_matrix_sympy_expression_with_d_rows_and_get_column_dimension(
        matrix: SympyMatrixCandidate,
        d: int,
        attempt_fix: bool = True,
        label: str = "sympy matrix") -> tp.Tuple[sym.MatrixBase, int]:
    matrix = check_sympy_matrix(matrix=matrix,
                                attempt_fix=attempt_fix,
                                label=label)

    if matrix.shape[0] != d:
        raise ValueError(f"{label} must be a matrix with {d} rows but has "
                         f"{matrix.shape[0]} rows")
    else:
        n = matrix.shape[1]
        return matrix, n


def check_matrix_sympy_expression_with_d_columns_and_get_row_dimension(
        matrix: SympyMatrixCandidate,
        d: int,
        attempt_fix: bool = True,
        label: str = "sympy matrix") -> tp.Tuple[sym.MatrixBase, int]:
    matrix = check_sympy_matrix(matrix=matrix,
                                attempt_fix=attempt_fix,
                                label=label)

    if matrix.shape[1] != d:
        raise ValueError(f"{label} must be a matrix with {d} rows but has "
                         f"{matrix.shape[1]} rows")
    else:
        n = matrix.shape[0]
        return matrix, n


def check_m_by_n_matrix_sympy_expression(
        matrix: SympyMatrixCandidate,
        m: int,
        n: int,
        attempt_fix: bool = True,
        label: str = 'sympy matrix') -> sym.MatrixBase:
    matrix = check_sympy_matrix(matrix=matrix,
                                attempt_fix=attempt_fix,
                                label=label)

    if matrix.rows == m and matrix.cols == n:
        return matrix
    else:
        raise ValueError(f"{label} must be a {m}x{n} matrix but is "
                         f"{matrix.rows}x{matrix.cols} dimensional")


def check_d_dimensional_square_matrix_sympy_expression(
        matrix: SympyMatrixCandidate,
        d: int,
        attempt_fix: bool = True,
        label: str = "sympy matrix") -> sym.MatrixBase:
    return check_m_by_n_matrix_sympy_expression(matrix=matrix,
                                                m=d,
                                                n=d,
                                                attempt_fix=attempt_fix,
                                                label=label)


def check_d_dimensional_column_vector_sympy_expression(
        matrix: SympyMatrixCandidate,
        d: int,
        attempt_fix: bool = True,
        label: str = "sympy matrix") -> sym.MatrixBase:
    return check_m_by_n_matrix_sympy_expression(matrix=matrix,
                                                m=d,
                                                n=1,
                                                attempt_fix=attempt_fix,
                                                label=label)


def check_d_dimensional_row_vector_sympy_expression(
        matrix: SympyMatrixCandidate,
        d: int,
        attempt_fix: bool = True,
        label: str = "sympy matrix") -> sym.MatrixBase:
    return check_m_by_n_matrix_sympy_expression(matrix=matrix,
                                                m=1,
                                                n=d,
                                                attempt_fix=attempt_fix,
                                                label=label)
