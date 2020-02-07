import numbers
import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import sympy as sym


class ParameterTransformation(ABC):
    @abstractmethod
    def transform_params(self, unconstrained: np.ndarray):
        pass

    @abstractmethod
    def untransform_params(self, constrained: np.ndarray):
        pass


class UnivariateTransformation(ABC):
    @abstractmethod
    def transform_param(self, unconstrained: numbers.Number):
        pass

    @abstractmethod
    def untransform_param(self, constrained: numbers.Number):
        pass


class LambdaUnivariateTransformation(UnivariateTransformation):
    def __init__(self,
                 transform_function: tp.Callable[[numbers.Number],
                                                 numbers.Number],
                 untransform_function: tp.Callable[[numbers.Number],
                                                   numbers.Number]):

        self._transform_function = transform_function
        self._untransform_function = untransform_function

    def transform_param(self, unconstrained: numbers.Number):
        return self._transform_function(unconstrained)

    @abstractmethod
    def untransform_param(self, constrained: numbers.Number):
        return self._untransform_function(constrained)


class IndependentParameterTransformation(ParameterTransformation):
    def __init__(self,
                 parameter_symbols: tp.Tuple[sym.Symbol, ...],
                 parameter_symbol_to_univariate_transformation_map:
                 tp.Dict[sym.Symbol, UnivariateTransformation]):
        self._parameter_symbols = parameter_symbols
        self._parameter_symbol_to_univariate_transformation_map \
            = parameter_symbol_to_univariate_transformation_map

    def transform_params(self, unconstrained: np.ndarray):
        constrained = np.full_like(unconstrained, fill_value=np.nan)

        for i, parameter_symbol in enumerate(self._parameter_symbols):
            univariate_transform = \
                (self
                 ._parameter_symbol_to_univariate_transformation_map
                 .get(parameter_symbol))

            if univariate_transform is None:
                constrained[i] = unconstrained[i]
            else:
                constrained[i] = univariate_transform.transform_param(unconstrained[i])

        return constrained

    def untransform_params(self, constrained: np.ndarray):
        unconstrained = np.full_like(constrained, fill_value=np.nan)

        for i, parameter_symbol in enumerate(self._parameter_symbols):
            univariate_transform = \
                (self
                 ._parameter_symbol_to_univariate_transformation_map
                 .get(parameter_symbol))

            if univariate_transform is None:
                unconstrained[i] = constrained[i]
            else:
                unconstrained[i] = univariate_transform.untransform_param(constrained[i])

        return unconstrained


# @dataclass(frozen=True)
# class UnivariateBounds:
#     upper: numbers.Number
#     lower: numbers.Number
#
#
# class RectangularParameterRestriction(IndependentParameterTransformation):
#     def __init__(self,
#                  parameter_symbols: tp.Tuple[sym.Symbol, ...],
#                  parameter_symbol_to_bounds_map: tp.Dict[sym.Symbol,
#                                                          UnivariateBounds]):
#         self._parameter_symbols = parameter_symbols
#         self._parameter_symbol_to_bounds_map = parameter_symbol_to_bounds_map
