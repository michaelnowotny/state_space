import numpy as np
import sympy as sym

from state_space import (
    SymbolicStateSpaceModelViaMaximumLikelihood,
    LambdaUnivariateTransformation,
    IndependentParameterTransformation
)


class SymbolicTimeVaryingEquityPremiumModel(
        SymbolicStateSpaceModelViaMaximumLikelihood):
    def __init__(self, excess_returns: np.ndarray):
        parameter_symbols = \
            sym.symbols(('alpha.mu',
                         'beta.mu',
                         'sigma2',
                         'sigma2.mu'))

        alpha_mu, beta_mu, sigma_2, sigma_2_mu = parameter_symbols

        H = sym.Matrix([sigma_2])
        Q = sym.Matrix([sigma_2_mu])
        c = sym.Matrix([alpha_mu])
        T = sym.Matrix([beta_mu])
        Z = sym.Matrix.ones(rows=1, cols=1)
        R = sym.Matrix.ones(rows=1, cols=1)

        state_vector_symbols = tuple([sym.Symbol('mu')])
        y = sym.Symbol('y')
        data_symbol_to_data_map = {y: excess_returns}
        excess_return_variance = np.var(excess_returns)
        parameter_symbols_to_start_parameters_map = \
            {alpha_mu: 0.0,
             beta_mu: 0.0,
             sigma_2: excess_return_variance,
             sigma_2_mu: excess_return_variance}

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
                         observation_vector_symbols=tuple([y]),
                         data_symbol_to_data_map=data_symbol_to_data_map,
                         parameter_symbols_to_start_parameters_map
                         =parameter_symbols_to_start_parameters_map,
                         parameter_transformation=parameter_transformation,
                         design_matrix=Z,
                         selection_matrix=R,
                         transition_matrix=T,
                         state_intercept_vector=c,
                         state_covariance_matrix=Q,
                         observation_covariance_matrix=H)


time_varying_equity_premium_model = \
    SymbolicTimeVaryingEquityPremiumModel(excess_returns=np.ones((15, )))