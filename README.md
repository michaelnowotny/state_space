# State Space
State Space is a Python package that permits the symbolic specification of linear state space models with normally distributed innovations and measurement errors. 
Coefficients are defined via SymPy matrices which are then compiled into a numerical statsmodels implementation.
These unobserved state is inferred via Kalman filtering and model parameters are estimated via maximum likelihood using statsmodels as the numerical backend.

## Stochastic Process
A linear state space model consists of a state evolution equation and an observation equation. 
The state is not directly observed, instead a linear transformation of the state with added Gaussian noise is observed. 
In a linear state space model, the state evolves according to

![State Transition Equation](https://raw.githubusercontent.com/michaelnowotny/state_space/master/images/state_transition_equation.png),

where the coefficients T, c, and R may depend on exogenous variables but not on the state itself. 
They may involve parameters that must be estimated from the data. 
The state innovation \eta_t has a multivariate normal distribution with zero mean and covariance matrix Q, which may depend on exogenous variables but not the state itself.

The observation equation maps the unobserved state according to

![Observation Equation](https://raw.githubusercontent.com/michaelnowotny/state_space/master/images/observation_equation.png).

The observation noise \epsilon_t has a multivariate normal distribution with zero mean and covariance matrix H. 
The coefficients Z, d, as well as H may depend on exogeneous data and involve unknown parameters that are estimated via MLE.

State Space adopts the following terminology:  
        T: transition matrix  
        c: state intercept vector  
        R: selection matrix  
        Q: state covariance matrix  
        Z: design matrix  
        d: observation intercept  
        H: observation covariance matrix  

The coefficients T,c, R, Q, Z, d, H are specified as SymPy matrices and may involve unknown parameters and exogenous data.

## Examples
State Space includes two examples in Jupyter notebooks:  
1.) A [conditional linear factor model](notebooks/Conditional%20Linear%20Factor%20Model.ipynb) for returns of the Ford motor corporation with S&P 500 returns as the factor.  
2.) A model of [time-variation in the equity premium](notebooks/Time-Variation%20in%20the%20Equity%20Premium.ipynb) applied to S&P 500 index data.  

## Installation
<pre>
    pip install state_space
</pre>
or 
<pre>
    pip3 install state_space
</pre>
if not using Anaconda.
