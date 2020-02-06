from abc import ABC, abstractmethod
from dataclasses import dataclass
import numbers
import numpy as np
import statsmodels as sm
import sympy as sym
import typing as tp

from .symbolic_dimension_checks import (
    SympyMatrixCandidate,
    check_sympy_matrix,
    check_d_dimensional_square_matrix_sympy_expression,
    check_d_dimensional_column_vector_sympy_expression,
    check_m_by_n_matrix_sympy_expression)


def _symbols_in_expression(
        symbols: tp.Tuple[sym.Symbol, ...],
        expression: tp.Union[sym.Expr, sym.MatrixBase, sym.Array]):
    return any([symbol in expression.free_symbols for symbol in symbols])


@dataclass(frozen=True)
class SymbolicStateSpaceModelCoefficients:
    Z: SympyMatrixCandidate
    H: SympyMatrixCandidate
    R: SympyMatrixCandidate
    Q: SympyMatrixCandidate
    T: tp.Optional[SympyMatrixCandidate] = None
    c: tp.Optional[SympyMatrixCandidate] = None
    d: tp.Optional[SympyMatrixCandidate] = None


def _check_and_fix_input_matrices_and_infer_dimensions(
        coefficients: SymbolicStateSpaceModelCoefficients,
        attempt_fix: bool = True) \
        -> tp.Tuple[SymbolicStateSpaceModelCoefficients, int, int, int]:
    Z = check_sympy_matrix(matrix=coefficients.Z,
                           attempt_fix=attempt_fix,
                           label='Z')

    R = check_sympy_matrix(matrix=coefficients.R,
                           attempt_fix=attempt_fix,
                           label='R')

    k_endog = Z.shape[0]
    k_states = Z.shape[1]
    k_posdef = R.shape[1]

    if R.shape[0] != k_states:
        raise ValueError('The number of rows of R must correspond to the '
                         'number of states (number of columns of Z).')

    if coefficients.T is not None:
        T = check_d_dimensional_square_matrix_sympy_expression(
                matrix=coefficients.T,
                d=k_states,
                attempt_fix=attempt_fix,
                label='T')
    else:
        T = None

    if coefficients.c is not None:
        c = check_d_dimensional_column_vector_sympy_expression(
                matrix=coefficients.c,
                d=k_states,
                attempt_fix=attempt_fix,
                label='c')
    else:
        c = None

    if coefficients.Q is not None:
        Q = check_d_dimensional_square_matrix_sympy_expression(
                matrix=coefficients.Q,
                d=k_posdef,
                attempt_fix=attempt_fix,
                label='Q')
    else:
        Q = None

    if coefficients.d is not None:
        d = check_d_dimensional_column_vector_sympy_expression(
                matrix=coefficients.d,
                d=k_endog,
                attempt_fix=attempt_fix,
                label='d')
    else:
        d = None

    if coefficients.H is not None:
        H = check_d_dimensional_square_matrix_sympy_expression(
                matrix=coefficients.H,
                d=k_endog,
                attempt_fix=attempt_fix,
                label='H')
    else:
        H = None

    return (SymbolicStateSpaceModelCoefficients(Z=Z,
                                                R=R,
                                                T=T,
                                                c=c,
                                                Q=Q,
                                                d=d,
                                                H=H),
            k_endog,
            k_states,
            k_posdef)


def _ensure_symbols_not_in_coefficients(
        state_vector_symbols: tp.Tuple[sym.Symbol, ...],
        coefficients: SymbolicStateSpaceModelCoefficients):
    for coefficient in (coefficients.T,
                        coefficients.c,
                        coefficients.d,
                        coefficients.H,
                        coefficients.Q,
                        coefficients.R,
                        coefficients.Z):
        if _symbols_in_expression(symbols=state_vector_symbols,
                                  expression=coefficient):
            raise ValueError(f'State vector symbol(s) must not appear in any '
                             f'coefficient, but they do in {coefficient.name}.')


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


class SymbolicStateSpaceModelViaMaximumLikelihood(sm.tsa.statespace.MLEModel):
    """
    This class models a linear state space model. The unobserved state evolves
    according to

                \alpha_{t+1} &  T_t \alpha_t + c_t + R_t \eta_t

    and

    The dynamics of the observed variables is given by

                y_t = Z_t \alpha_t + d_t + \epsilon_t.

    The distributions of state innovations \eta_t and measurement errors
    \epsilon_t are i.i.d. normal with

                \eta_t ~ N(0, Q_t)

    and

                \epsilon_t ~ N(0, H_t).

    The vector of observations y_t has dimension k_endog (endogenous variables).
    The vector of states \alpha_t has dimension k_states (unobserved variables).

    The coefficients of the system (T, c, R, Q, Z, d, H) may depend on exogenous
    variables.

    Terminology:
        T: transition matrix
        c: state intercept vector
        R: selection matrix
        Q: state covariance matrix
        Z: design matrix
        d: observation intercept
        H: observation covariance matrix


    """

    def __init__(self,
                 parameter_symbols: tp.Tuple[sym.Symbol, ...],
                 state_vector_symbols: tp.Tuple[sym.Symbol, ...],
                 data_symbol_to_data_map: tp.Dict[sym.Symbol, np.ndarray],
                 parameter_symbols_to_start_parameters_map:
                 tp.Dict[sym.Symbol, numbers.Number],
                 parameter_transformation: ParameterTransformation,
                 Z: SympyMatrixCandidate,
                 H: SympyMatrixCandidate,
                 R: SympyMatrixCandidate,
                 Q: SympyMatrixCandidate,
                 T: tp.Optional[SympyMatrixCandidate] = None,
                 c: tp.Optional[SympyMatrixCandidate] = None,
                 d: tp.Optional[SympyMatrixCandidate] = None):
        """
        T: a symbolic matrix with dimension k_states x k_states
        c: a symbolic vector with dimension k_states
        R: a symbolic matrix with dimension k_states x k_posdef
        Q: a positive definite symbolic covariance matrix with dimension k_posdef x k_posdef
        Z: a symbolic matrix with dimension k_endog x k_states
        d: a symbolic vector with dimension k_endog
        H: a positive definite symbolic covariance matrix with dimension k_endog x k_endog
        k_posdef: the dimensionality of the random vector of shocks to the state
        """
        self._parameter_symbols = parameter_symbols
        self._parameter_symbols_to_start_parameters_map = \
            parameter_symbols_to_start_parameters_map
        self._parameter_transformation = parameter_transformation

        # check coefficients and infer model dimension
        self._coefficients, self._k_endog, self._k_states, self._k_posdef = \
            _check_and_fix_input_matrices_and_infer_dimensions(
                SymbolicStateSpaceModelCoefficients(Z=Z,
                                                    R=R,
                                                    T=T,
                                                    c=c,
                                                    Q=Q,
                                                    d=d,
                                                    H=H))

        # make sure that the coefficients do not contain the state
        _ensure_symbols_not_in_coefficients(
            state_vector_symbols=state_vector_symbols,
            coefficients=self._coefficients)

        # check to make sure that the dimension of the state vector matches the
        # dimension of the state transition matrix
        if not len(state_vector_symbols) == self.k_states:
            raise ValueError('The dimension of the state vector must match the '
                             'dimension of the state transition matrix.')

        # infer the number of observations
        n_obs = max([len(x) for x in data_symbol_to_data_map.values()])

        # construct endogenous data (vector of observations)
        endogenous_data = np.full((self.k_states, n_obs), fill_value=np.nan)
        for i, state_vector_symbol in enumerate(state_vector_symbols):
            endogenous_data[i, :] = data_symbol_to_data_map[state_vector_symbol]

        # Initialize the numeric state space representation
        (super(SymbolicStateSpaceModelViaMaximumLikelihood, self)
         .__init__(endog=endogenous_data,
                   k_endog=self.k_endog,
                   k_posdef=self.k_posdef,
                   initialization='approximate_diffuse',
                   loglikelihood_burn=self.k_states))

        # The transition matrix must be of shape
        # (k_states, k_states, n_obs) if coefficients are time-varying or
        # (k_states, k_states) if coefficients are time-invariant
        transition_matrix = np.zeros((1, 1))

        # The design matrix must be of shape (k_endog, k_states, n_obs)
        design_matrix = np.ones((self.k_endog, self.k_states, n_obs))

        # Initialize the matrices
        self.ssm['design'] = design_matrix
        self.ssm['transition'] = transition_matrix
        self.ssm['selection'] = np.ones((1, 1))

        # Cache some indices

    #         self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

    # Exogenous data
    #         (self.k_exog, exog) = prepare_exog(exog)

    @property
    def coefficients(self) -> SymbolicStateSpaceModelCoefficients:
        return self._coefficients

    @property
    def k_endog(self) -> int:
        return self._k_endog

    @property
    def k_states(self) -> int:
        return self._k_states

    @property
    def k_posdef(self) -> int:
        return self._k_posdef

    @property
    def parameter_symbols(self) -> tp.Tuple[sym.Symbol, ...]:
        return self._parameter_symbols

    @property
    def parameter_symbols_to_start_parameters_map(self) \
            -> tp.Dict[sym.Symbol, numbers.Number]:
        return self._parameter_symbols_to_start_parameters_map

    @property
    def param_names(self) -> tp.Tuple[str, ...]:
        return tuple([parameter_symbol.name
                      for parameter_symbol
                      in self.parameter_symbols])

    @property
    def start_params(self):
        start_parameters = \
            [self._parameter_symbols_to_start_parameters_map[parameter_symbol]
             for parameter_symbol
             in self.parameter_symbols]

        return np.array(start_parameters)

    def transform_params(self, unconstrained):
        return (self
                ._parameter_transformation
                .transform_params(unconstrained=unconstrained))

    def untransform_params(self, constrained):
        return (self
                ._parameter_transformation
                .untransform_params(constrained=constrained))

    def update(self, params, *args, **kwargs):
        # params \
        #     = (super(SymbolicStateSpaceModelViaMaximumLikelihood, self)
        #        .update(params, *args, **kwargs))

        parameter_symbols_to_parameters_map = \
            {parameter_symbol: numeric_parameter
             for numeric_parameter, parameter_symbol
             in zip(params, self.parameter_symbols)}

        alpha_mu, beta_mu, sigma_2, sigma_2_mu = params

        # ToDo: Optimization for matrices that do not depend on parameters (put them in init)
        # Observation covariance
        # ToDo: loop over all elements
        self.ssm['obs_cov', 0, 0] = \
            self.H.eval(parameter_symbols_to_parameters_map)

        # State covariance
        self.ssm['state_cov', 0, 0] = \
            self.Q.eval(parameter_symbols_to_parameters_map)

        # State intercept
        if self.c is not None:
            self.ssm['state_intercept', 0, 0] = None

        # State transition
        self.ssm['transition', 0, 0] = beta_mu

        # self.ssm['design'] = design_matrix
        # self.ssm['selection'] = np.ones((1, 1))

        # self.ssm['obs_intercept', 0, 0]



# ToDo: Q and H must not be None

class SymbolicTimeVaryingEquityPremiumModel(
        SymbolicStateSpaceModelViaMaximumLikelihood):
    def __init__(self, excess_returns: np.ndarray):
        parameter_symbols = \
            sym.symbols(('alpha.mu',
                         'beta.mu',
                         'sigma2',
                         'sigma2.mu'))

        alpha_mu, beta_mu, sigma_2, sigma_2_mu = parameter_symbols

        H = sym.Matrix(sigma_2)
        Q = sym.Matrix(sigma_2_mu)
        c = sym.Matrix(alpha_mu)
        T = sym.Matrix(beta_mu)
        Z = sym.Matrix.ones(rows=1, cols=1)
        R = sym.Matrix.ones(rows=1, cols=1)

        state_vector_symbols = sym.symbols('mu')
        y = sym.Symbol('y')
        data_symbol_to_data_map = {y: excess_returns}
        parameter_symbols_to_start_parameters_map = \
            {alpha_mu: 0.0,
             beta_mu: 0.0,
             sigma_2: np.var(self.endog),
             sigma_2_mu: np.var(self.endog)}

        squared_univariate_transform = \
            LambdaUnivariateTransformation(
                transform_function=lambda x: x**2,
                untransform_function=lambda x: x**0.5)

        parameter_transformation = \
            IndependentParameterTransformation(
                parameter_symbols=parameter_symbols,
                parameter_symbol_to_univariate_transformation_map
                ={sigma_2: squared_univariate_transform,
                  sigma_2_mu: squared_univariate_transform})

        super().__init__(parameter_symbols=parameter_symbols,
                         state_vector_symbols=state_vector_symbols,
                         data_symbol_to_data_map=data_symbol_to_data_map,
                         parameter_symbols_to_start_parameters_map
                         =parameter_symbols_to_start_parameters_map,
                         parameter_transformation=parameter_transformation,
                         Z=Z,
                         R=R,
                         T=T,
                         c=c,
                         Q=Q,
                         H=H)


