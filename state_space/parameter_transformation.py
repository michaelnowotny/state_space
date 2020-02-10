from abc import (
    ABC,
    abstractmethod
)
# from dataclasses import dataclass
import numbers
import numpy as np
import sympy as sym
import typing as tp


class ParameterTransformation(ABC):
    """
    This class models a parameter transformation from an unconstrained space in
    which the nunerical optimizer operates into a constrained space for the
    parameters of the model and vice versa via a transform_params method and an
    untransform_params method respectively.
    """
    @abstractmethod
    def transform_params(self, unconstrained: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def untransform_params(self, constrained: np.ndarray) -> np.ndarray:
        pass


class LambdaParameterTransformation(ParameterTransformation):
    """
    This class enables the specification of transformation and untransformation
    methods via functions passed to the constructor.
    """
    def __init__(self,
                 transform_function: tp.Callable[[np.ndarray],
                                                 np.ndarray],
                 untransform_function: tp.Callable[[np.ndarray],
                                                   np.ndarray]
                 ):
        self._transform_function = transform_function
        self._untransform_function = untransform_function

    def transform_params(self, unconstrained: np.ndarray) -> np.ndarray:
        return self._transform_function(unconstrained)

    def untransform_params(self, constrained: np.ndarray) -> np.ndarray:
        return self._untransform_function(constrained)


class UnivariateTransformation(ABC):
    """
    This class models a parameter transformation/untransformation of a single
    parameter.
    """
    @abstractmethod
    def transform_param(self, unconstrained: numbers.Number) -> numbers.Number:
        pass

    @abstractmethod
    def untransform_param(self, constrained: numbers.Number) -> numbers.Number:
        pass


class LambdaUnivariateTransformation(UnivariateTransformation):
    """
    This class enables the specification of transformation and untransformation
    methods for a single parameter via functions passed to the constructor.
    """
    def __init__(self,
                 transform_function: tp.Callable[[numbers.Number],
                                                 numbers.Number],
                 untransform_function: tp.Callable[[numbers.Number],
                                                   numbers.Number]):

        self._transform_function = transform_function
        self._untransform_function = untransform_function

    def transform_param(self, unconstrained: numbers.Number) -> numbers.Number:
        return self._transform_function(unconstrained)

    def untransform_param(self, constrained: numbers.Number) -> numbers.Number:
        return self._untransform_function(constrained)


class IndependentParameterTransformation(ParameterTransformation):
    """
    This class models a parameter transformations where individual parameters
    are independent of each other.
    """
    def __init__(self,
                 parameter_symbols: tp.Tuple[sym.Symbol, ...],
                 parameter_symbol_to_univariate_transformation_map:
                 tp.Dict[sym.Symbol, UnivariateTransformation]):
        self._parameter_symbols = parameter_symbols
        self._parameter_symbol_to_univariate_transformation_map \
            = parameter_symbol_to_univariate_transformation_map

    def transform_params(self, unconstrained: np.ndarray) -> np.ndarray:
        constrained = np.full_like(unconstrained, fill_value=np.nan)

        for i, parameter_symbol in enumerate(self._parameter_symbols):
            univariate_transform = \
                (self
                 ._parameter_symbol_to_univariate_transformation_map
                 .get(parameter_symbol))

            if univariate_transform is None:
                constrained[i] = unconstrained[i]
            else:
                constrained[i] \
                    = univariate_transform.transform_param(unconstrained[i])

        return constrained

    def untransform_params(self, constrained: np.ndarray) -> np.ndarray:
        unconstrained = np.full_like(constrained, fill_value=np.nan)

        for i, parameter_symbol in enumerate(self._parameter_symbols):
            univariate_transform = \
                (self
                 ._parameter_symbol_to_univariate_transformation_map
                 .get(parameter_symbol))

            if univariate_transform is None:
                unconstrained[i] = constrained[i]
            else:
                unconstrained[i] \
                    = univariate_transform.untransform_param(constrained[i])

        return unconstrained


def _transform_negative_inf_positive_inf_to_a_b(x: float,
                                                a: float,
                                                b: float) \
        -> float:
    return a + (b - a) / (1 + np.exp(-x))


def _transform_a_b_to_negative_inf_positive_inf(x: float,
                                                a: float,
                                                b: float) \
        -> float:
    if a >= b:
        raise ValueError('a must be less than b')
    if x <= a:
        return -np.inf
    if x >= b:
        return np.inf

    return np.log((x - a) / (b - x))


class UpperAndLowerBoundUnivariateParameterTransformation(LambdaUnivariateTransformation):
    def __init__(self,
                 upper: float,
                 lower: float):
        super().__init__(transform_function=lambda x: _transform_negative_inf_positive_inf_to_a_b(x=x,
                                                                                                 a=lower,
                                                                                                 b=upper),
                        untransform_function=lambda x: _transform_a_b_to_negative_inf_positive_inf(x=x,
                                                                                                   a=lower,
                                                                                                   b=upper))


def _transform_negative_inf_positive_inf_to_a_positive_inf(x: float,
                                                           a: float) -> float:
    return a + np.exp(x)


def _transform_a_positive_inf_to_negative_inf_positive_inf(x: float,
                                                           a: float) -> float:
    return np.log(x - a)


class LowerBoundUnivariateParameterTransformation(LambdaUnivariateTransformation):
    def __init__(self,
                 lower: float):
        super().__init__(transform_function=lambda x: _transform_negative_inf_positive_inf_to_a_positive_inf(x=x,
                                                                                                             a=lower),
                        untransform_function=lambda x: _transform_a_positive_inf_to_negative_inf_positive_inf(x=x,
                                                                                                              a=lower))


def _transform_negative_inf_positive_inf_to_negative_inf_b(x: float,
                                                           b: float) -> float:
    return b - np.exp(-x)


def _transform_negative_inf_b_to_negative_inf_positive_inf(x: float,
                                                           b: float) -> float:
    return - np.log(b - x)


class UpperBoundUnivariateParameterTransformation(LambdaUnivariateTransformation):
    def __init__(self,
                 upper: float):
        super().__init__(transform_function=lambda x: _transform_negative_inf_positive_inf_to_negative_inf_b(x=x,
                                                                                                             b=upper),
                        untransform_function=lambda x: _transform_negative_inf_b_to_negative_inf_positive_inf(x=x,
                                                                                                              b=upper))


class UnivariateIdentityTransformation(LambdaUnivariateTransformation):
    def __init__(self):
        super().__init__(transform_function=lambda x: x,
                         untransform_function=lambda x: x)


class RectangularParameterTransformation(IndependentParameterTransformation):
    """
    This class models a rectangular parameter space where a tuple of the upper
    and lower bound is specified for each constrained parameter.
    """
    def __init__(self,
                 parameter_symbols: tp.Tuple[sym.Symbol, ...],
                 parameter_symbol_to_bounds_map: tp.Dict[sym.Symbol, tp.Tuple[float, float]]):
        self._parameter_symbols = parameter_symbols
        self._parameter_symbol_to_bounds_map = parameter_symbol_to_bounds_map

        parameter_symbol_to_univariate_transformation_map: \
            tp.Dict[sym.Symbol, UnivariateTransformation] = {}

        for parameter_symbol, (lower, upper) in parameter_symbol_to_bounds_map.items():
            if np.isfinite(lower) and np.isfinite(upper):
                univariate_transformation = \
                    UpperAndLowerBoundUnivariateParameterTransformation(lower=lower,
                                                                        upper=upper)
            elif np.isfinite(lower):
                univariate_transformation = \
                    LowerBoundUnivariateParameterTransformation(lower=lower)
            elif np.isfinite(upper):
                univariate_transformation = \
                    UpperBoundUnivariateParameterTransformation(upper=upper)
            else:
                univariate_transformation = UnivariateIdentityTransformation()

            parameter_symbol_to_univariate_transformation_map[parameter_symbol] = \
                univariate_transformation

        super().__init__(parameter_symbols=parameter_symbols,
                         parameter_symbol_to_univariate_transformation_map
                         =parameter_symbol_to_univariate_transformation_map)

