from dataclasses import dataclass
import numbers
import numpy as np
import statsmodels.api as sm
import sympy as sym
import typing as tp

from state_space.compiled_matrix import CompiledMatrix
from state_space.parameter_transformation import (
    ParameterTransformation
)

from state_space.symbolic_dimension_checks import (
    SympyMatrixCandidate,
    check_sympy_matrix,
    check_d_dimensional_square_matrix_sympy_expression,
    check_d_dimensional_column_vector_sympy_expression
)


def _symbols_in_expression(
        symbols: tp.Tuple[sym.Symbol, ...],
        expression: tp.Union[sym.Expr, sym.MatrixBase, sym.Array]):
    return any([symbol in expression.free_symbols for symbol in symbols])


@dataclass(frozen=True)
class SymbolicStateSpaceModelCoefficients:
    design_matrix: SympyMatrixCandidate
    observation_covariance_matrix: SympyMatrixCandidate
    selection_matrix: SympyMatrixCandidate
    state_covariance_matrix: SympyMatrixCandidate
    transition_matrix: tp.Optional[SympyMatrixCandidate] = None
    state_intercept_vector: tp.Optional[SympyMatrixCandidate] = None
    observation_intercept_vector: tp.Optional[SympyMatrixCandidate] = None

    @property
    def stats_models_coefficient_label_to_symbolic_coefficient_map(self) \
            -> tp.Dict[str, sym.MatrixBase]:
        return {'design': self.design_matrix,
                'obs_intercept': self.observation_intercept_vector,
                'obs_cov': self.observation_covariance_matrix,
                'transition': self.transition_matrix,
                'state_intercept': self.state_intercept_vector,
                'selection': self.selection_matrix,
                'state_cov': self.state_covariance_matrix}

    @property
    def coefficients(self) -> tp.Tuple[tp.Optional[SympyMatrixCandidate], ...]:
        return (self.design_matrix,
                self.observation_covariance_matrix,
                self.selection_matrix,
                self.state_covariance_matrix,
                self.transition_matrix,
                self.state_intercept_vector,
                self.observation_intercept_vector)

    @property
    def free_symbols(self) -> tp.FrozenSet[sym.Symbol]:
        return frozenset().union(*[coefficient.free_symbols
                                   for coefficient
                                   in self.coefficients
                                   if coefficient is not None])


def _check_and_fix_input_matrices_and_infer_dimensions(
        coefficients: SymbolicStateSpaceModelCoefficients,
        attempt_fix: bool = True) \
        -> tp.Tuple[SymbolicStateSpaceModelCoefficients, int, int, int]:
    design_matrix = check_sympy_matrix(matrix=coefficients.design_matrix,
                                       attempt_fix=attempt_fix,
                                       label='design_matrix')

    selection_matrix = check_sympy_matrix(matrix=coefficients.selection_matrix,
                                          attempt_fix=attempt_fix,
                                          label='selection_matrix')

    k_endog = design_matrix.shape[0]
    k_states = design_matrix.shape[1]
    k_posdef = selection_matrix.shape[1]

    if selection_matrix.shape[0] != k_states:
        raise ValueError('The number of rows of selection_matrix must correspond to the '
                         'number of states (number of columns of design_matrix).')

    if coefficients.transition_matrix is not None:
        transition_matrix = check_d_dimensional_square_matrix_sympy_expression(
                matrix=coefficients.transition_matrix,
                d=k_states,
                attempt_fix=attempt_fix,
                label='transition_matrix')
    else:
        transition_matrix = None

    if coefficients.state_intercept_vector is not None:
        state_intercept_vector = \
            check_d_dimensional_column_vector_sympy_expression(
                matrix=coefficients.state_intercept_vector,
                d=k_states,
                attempt_fix=attempt_fix,
                label='state_intercept_vector')
    else:
        state_intercept_vector = None

    if coefficients.state_covariance_matrix is not None:
        state_covariance_matrix = \
            check_d_dimensional_square_matrix_sympy_expression(
                matrix=coefficients.state_covariance_matrix,
                d=k_posdef,
                attempt_fix=attempt_fix,
                label='state_covariance_matrix')
    else:
        state_covariance_matrix = None

    if coefficients.observation_intercept_vector is not None:
        observation_intercept_vector = \
            check_d_dimensional_column_vector_sympy_expression(
                matrix=coefficients.observation_intercept_vector,
                d=k_endog,
                attempt_fix=attempt_fix,
                label='observation_intercept_vector')
    else:
        observation_intercept_vector = None

    if coefficients.observation_covariance_matrix is not None:
        observation_covariance_matrix = \
            check_d_dimensional_square_matrix_sympy_expression(
                matrix=coefficients.observation_covariance_matrix,
                d=k_endog,
                attempt_fix=attempt_fix,
                label='observation_covariance_matrix')
    else:
        observation_covariance_matrix = None

    return (SymbolicStateSpaceModelCoefficients(
                design_matrix=design_matrix,
                selection_matrix=selection_matrix,
                transition_matrix=transition_matrix,
                state_intercept_vector=state_intercept_vector,
                state_covariance_matrix=state_covariance_matrix,
                observation_intercept_vector=observation_intercept_vector,
                observation_covariance_matrix=observation_covariance_matrix),
            k_endog,
            k_states,
            k_posdef)


def _ensure_symbols_not_in_coefficients(
        state_vector_symbols: tp.Tuple[sym.Symbol, ...],
        coefficients: SymbolicStateSpaceModelCoefficients):
    for coefficient in (coefficients.transition_matrix,
                        coefficients.state_intercept_vector,
                        coefficients.observation_intercept_vector,
                        coefficients.observation_covariance_matrix,
                        coefficients.state_covariance_matrix,
                        coefficients.selection_matrix,
                        coefficients.design_matrix):
        if coefficient is not None and \
           _symbols_in_expression(symbols=state_vector_symbols,
                                  expression=coefficient):
            raise ValueError(f'State vector symbol(s) must not appear in any '
                             f'coefficient, but they do in {coefficient.name}.')


class SymbolicStateSpaceModelViaMaximumLikelihood(sm.tsa.statespace.MLEModel):
    """
    This class models a linear state space model. The unobserved state evolves
    according to

                s_{t+1} = T_t s_t + c_t + R_t \eta_t

    and

    The dynamics of the observed variables is given by

                y_t = Z_t s_t + d_t + \epsilon_t.

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

    def __init__(
            self,
            parameter_symbols: tp.Tuple[sym.Symbol, ...],
            state_vector_symbols: tp.Tuple[sym.Symbol, ...],
            observation_vector_symbols: tp.Tuple[sym.Symbol, ...],
            data_symbol_to_data_map: tp.Dict[sym.Symbol, np.ndarray],
            parameter_symbols_to_start_parameters_map:
            tp.Dict[sym.Symbol, numbers.Number],
            parameter_transformation: ParameterTransformation,
            design_matrix: SympyMatrixCandidate,
            observation_covariance_matrix: SympyMatrixCandidate,
            selection_matrix: SympyMatrixCandidate,
            state_covariance_matrix: SympyMatrixCandidate,
            transition_matrix: tp.Optional[SympyMatrixCandidate] = None,
            state_intercept_vector: tp.Optional[SympyMatrixCandidate] = None,
            observation_intercept_vector: tp.Optional[SympyMatrixCandidate] = None):
        """
        
        Args:
            parameter_symbols: 
            state_vector_symbols: 
            observation_vector_symbols: 
            data_symbol_to_data_map: 
            parameter_symbols_to_start_parameters_map: 
            parameter_transformation: 
            design_matrix: a symbolic matrix with dimension k_endog x k_states
            observation_covariance_matrix: a positive definite symbolic 
                                           covariance matrix with dimension 
                                           k_endog x k_endog
            selection_matrix: a symbolic matrix with dimension 
                              k_states x k_posdef
            state_covariance_matrix: a positive definite symbolic covariance 
                                     matrix with dimension k_posdef x k_posdef
            transition_matrix: a symbolic matrix with dimension 
                               k_states x k_states
            state_intercept_vector: a symbolic vector with dimension k_states
            observation_intercept_vector: a symbolic vector with dimension 
                                          k_endog
        """

        self._parameter_symbols = parameter_symbols
        self._parameter_symbols_to_start_parameters_map = \
            parameter_symbols_to_start_parameters_map
        self._parameter_transformation = parameter_transformation

        # check coefficients and infer model dimension
        self._coefficients, k_endog, k_states, k_posdef = \
            _check_and_fix_input_matrices_and_infer_dimensions(
                SymbolicStateSpaceModelCoefficients(
                    design_matrix=design_matrix,
                    selection_matrix=selection_matrix,
                    transition_matrix=transition_matrix,
                    state_intercept_vector=state_intercept_vector,
                    state_covariance_matrix=state_covariance_matrix,
                    observation_intercept_vector=observation_intercept_vector,
                    observation_covariance_matrix=observation_covariance_matrix))

        # make sure that the coefficients do not contain the state
        _ensure_symbols_not_in_coefficients(
            state_vector_symbols=state_vector_symbols,
            coefficients=self._coefficients)

        # check to make sure that the dimension of the state vector matches the
        # dimension of the state transition matrix
        if not len(state_vector_symbols) == k_states:
            raise ValueError('The dimension of the state vector must match the '
                             'dimension of the state transition matrix.')

        # check to make sure that the dimension of the observation vector
        # matches the dimension in the coefficients
        if not len(observation_vector_symbols) == k_endog:
            raise ValueError('The dimension of the observation vector must '
                             'match the dimension in the coefficients.')

        # infer the number of observations
        n_obs = max([len(x) for x in data_symbol_to_data_map.values()])

        # construct endogenous data (vector of observations)
        endogenous_data = np.full((k_endog, n_obs), fill_value=np.nan)
        for i, observation_vector_symbol in enumerate(observation_vector_symbols):
            endogenous_data[i, :] = data_symbol_to_data_map[observation_vector_symbol]

        endogenous_data = endogenous_data.squeeze()

        # Initialize the numeric state space representation
        (super(SymbolicStateSpaceModelViaMaximumLikelihood, self)
         .__init__(endog=endogenous_data,
                   # k_endog=k_endog,
                   k_states=k_states,
                   k_posdef=k_posdef,
                   initialization='approximate_diffuse',
                   loglikelihood_burn=k_states))

        # determine which symbols in coefficients are not parameters
        self._exogenous_data_symbols \
            = tuple(self
                    ._coefficients
                    .free_symbols
                    .difference(parameter_symbols))

        # organize exogenous data which the coefficients depend on in a tuple
        self._exogenous_data = \
            tuple([data_symbol_to_data_map[exogenous_data_symbol]
                   for exogenous_data_symbol
                   in self._exogenous_data_symbols])

        # link parameter symbols and exogenous data symbols
        all_parameter_symbols = \
            tuple(list(parameter_symbols) + list(self._exogenous_data_symbols))

        # compile coefficient matrices
        self._stats_models_coefficient_label_to_compiled_coefficient_map: \
            tp.Dict[str, CompiledMatrix] = \
            {label: CompiledMatrix(symbols=all_parameter_symbols,
                                   matrix_expression=coefficient,
                                   label=label)
             for label, coefficient
             in (self
                 ._coefficients
                 .stats_models_coefficient_label_to_symbolic_coefficient_map
                 .items())
             if coefficient is not None}

        # evaluate compiled coefficient matrices and populate statsmodels
        start_parameter_values_and_exogenous_data = \
            tuple(list(self.start_params) + list(self._exogenous_data))

        for label, compiled_coefficient \
                in (self
                    ._stats_models_coefficient_label_to_compiled_coefficient_map
                    .items()):
            a = (compiled_coefficient
                 .evaluate_matrix(numeric_values
                                  =start_parameter_values_and_exogenous_data))
            self.ssm[label] = a

    @property
    def coefficients(self) -> SymbolicStateSpaceModelCoefficients:
        return self._coefficients

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
    def exogenous_data_symbols(self) -> tp.Tuple[sym.Symbol, ...]:
        return self._exogenous_data_symbols

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

        # parameter_symbols_to_parameters_map = \
        #     {parameter_symbol: numeric_parameter
        #      for numeric_parameter, parameter_symbol
        #      in zip(params, self.parameter_symbols)}

        # evaluate compiled coefficient matrices and populate statsmodels
        numeric_values = \
            tuple(list(params) + list(self._exogenous_data))

        for label, compiled_coefficient \
                in (self
                    ._stats_models_coefficient_label_to_compiled_coefficient_map
                    .items()):
            if not compiled_coefficient.all_constant:
                (compiled_coefficient
                 .update_stats_models_matrix(ssm=self.ssm,
                                             numeric_values=numeric_values))
