import numbers
import numpy as np
import sympy as sym
import typing as tp


class CompiledMatrix:
    def __init__(self,
                 symbols: tp.Tuple[sym.Symbol, ...],
                 matrix_expression: sym.MatrixBase,
                 label: str):
        rows, cols = matrix_expression.shape

        compiled_elements = np.empty((rows, cols), dtype=np.dtype(object))
        is_constant = np.empty((rows, cols), dtype=np.bool)

        for i in rows:
            for j in cols:
                compiled_element = \
                    sym.lambdify(expr=matrix_expression[i, j])

                is_constant[i, j] = \
                    (len(matrix_expression
                         .free_symbols
                         .intersection(symbols))
                     == 0)

                compiled_elements[i, j] = compiled_element

        self._label = label
        self._compiled_elements = compiled_elements
        self._is_constant = is_constant
        self._all_constant = np.all(is_constant)
        self._rows = rows
        self._cols = cols

    @property
    def all_constant(self) -> bool:
        return self._all_constant

    def evaluate_matrix(
            self,
            numeric_values: tp.Tuple[tp.Union[numbers.Number,
                                              np.ndarray], ...]) \
            -> np.ndarray:
        evaluated_elements = \
            np.empty((self._rows, self._cols), dtype=np.dtype(object))
        sizes = np.zeros((self._rows, self._cols), dtype=np.int)

        for i in self._rows:
            for j in self._cols:
                evaluated_element = \
                    self._compiled_elements[i, j](numeric_values)
                sizes[i, j] = np.size(evaluated_element)
                evaluated_elements[i, j] = evaluated_element

        max_size = np.max(sizes)
        if max_size > 1:
            result = np.full((self._rows, self._cols, max_size),
                             fill_value=np.nan)
            for i in self._rows:
                for j in self._cols:
                    result[i, j, :] = evaluated_elements[i, j]
        else:
            result = np.full((self._rows, self._cols),
                             fill_value=np.nan)
            for i in self._rows:
                for j in self._cols:
                    result[i, j] = evaluated_elements[i, j]

        return result

    def set_stats_models_matrix(self,
                                ssm,
                                numeric_values: tp.Tuple[tp.Union[numbers.Number,
                                              np.ndarray], ...]):
        for i in self._rows:
            for j in self._cols:
                if not self._is_constant[i, j]:
                    evaluated_element = \
                        self._compiled_elements[i, j](numeric_values)
                    ssm[self._label, i, j] = evaluated_element
