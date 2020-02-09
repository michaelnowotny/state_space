import numpy as np
import sympy as sym

from state_space import (
    SymbolicStateSpaceModelViaMaximumLikelihood,
    LambdaParameterTransformation)


class SymbolicConditionalUnivariateLinearFactorModel(
    SymbolicStateSpaceModelViaMaximumLikelihood):
    def __init__(self,
                 security_excess_return: np.ndarray,
                 market_excess_return: np.ndarray):
        sigma_e_2, sigma_eta_2, sigma_epsilon_2, alpha, beta, r, r_M = \
            sym.symbols(
                'sigma_e_2, sigma_eta_2, sigma_epsilon_2, alpha, beta, r, r_M')

        parameter_symbols = (sigma_e_2, sigma_eta_2, sigma_epsilon_2)
        state_vector_symbols = (alpha, beta)
        observation_vector_symbols = (r,)
        data_symbol_to_data_map = {r: security_excess_return,
                                   r_M: market_excess_return}
        security_return_var = float(np.var(security_excess_return))
        parameter_symbols_to_start_parameters_map = \
            {sigma_e_2: security_return_var,
             sigma_eta_2: security_return_var,
             sigma_epsilon_2: security_return_var}

        parameter_transformation = \
            LambdaParameterTransformation(
                transform_function=lambda x: x ** 2,
                untransform_function=lambda x: x ** 0.5)

        transition_matrix = sym.eye(2)
        design_matrix = sym.Matrix([[1, r_M]])
        selection_matrix = sym.eye(2)
        state_covariance_matrix = \
            sym.diagonalize_vector(
            sym.Matrix([sigma_eta_2, sigma_epsilon_2]))

        observation_covariance_matrix = sym.Matrix([[sigma_e_2]])

        super().__init__(
                    parameter_symbols=parameter_symbols,
                    state_vector_symbols=state_vector_symbols,
                    observation_vector_symbols=observation_vector_symbols,
                    data_symbol_to_data_map=data_symbol_to_data_map,
                    parameter_symbols_to_start_parameters_map
                    =parameter_symbols_to_start_parameters_map,
                    parameter_transformation=parameter_transformation,
                    design_matrix=design_matrix,
                    observation_covariance_matrix=observation_covariance_matrix,
                    selection_matrix=selection_matrix,
                    state_covariance_matrix=state_covariance_matrix,
                    transition_matrix=transition_matrix)


if __name__ == '__main__':
    time_varying_equity_premium_model = \
        SymbolicConditionalUnivariateLinearFactorModel(
            security_excess_return=np.ones(15),
            market_excess_return=np.ones(15))
