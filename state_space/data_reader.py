import numpy as np
import pandas as pd
import typing as tp


def _multiply_columns_in_place(df: pd.DataFrame,
                               column_names: tp.Sequence[str],
                               factor: float):
    for column_name in column_names:
        df[column_name] = factor * df[column_name]


def _set_data_types(df: pd.DataFrame,
                    column_name_to_dtype_map: tp.Dict[str, str]) \
        -> pd.DataFrame:
    column_name_to_new_column_map = {}

    for column_name, dtype in column_name_to_dtype_map.items():
        column_name_to_new_column_map[column_name] \
            = df[column_name].astype(dtype)

    return pd.DataFrame(column_name_to_new_column_map, index=df.index)


def load_monthly_and_annual_ff_factors(filename: str) \
        -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    ff_factors \
        = pd.read_csv(filename,
                      engine='python',
                      skiprows=4,
                      skipfooter=1,
                      names=['Date', 'Mkt-RF', 'SMB', 'HML', 'RF'])

    for column_name in ('Date', 'Mkt-RF', 'SMB', 'HML', 'RF'):
        ff_factors[column_name] = ff_factors[column_name].str.strip()

    date_length = ff_factors['Date'].str.len()

    monthly_index = date_length == 6
    ff_monthly = ff_factors.loc[monthly_index, :]
    annual_index = date_length == 4
    ff_annual = ff_factors.loc[annual_index, :]

    new_monthly_index = pd.to_datetime(ff_monthly['Date'], format='%Y%m')
    new_annual_index = pd.to_datetime(ff_annual['Date'], format='%Y')

    ff_monthly.index = new_monthly_index
    ff_annual.index = new_annual_index

    # set data types
    column_name_to_dtype_map = {'Mkt-RF': 'float', 'SMB': 'float',
                                'HML': 'float', 'RF': 'float'}
    ff_monthly = _set_data_types(ff_monthly, column_name_to_dtype_map)
    ff_annual = _set_data_types(ff_annual, column_name_to_dtype_map)

    # rescale returns
    columns_to_rescale = ['Mkt-RF', 'SMB', 'HML', 'RF']
    _multiply_columns_in_place(ff_monthly, columns_to_rescale, 0.01)
    _multiply_columns_in_place(ff_annual, columns_to_rescale, 0.01)

    return ff_monthly, ff_annual


def compute_monthly_returns(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Must have date index
    """
    returns = np.exp(np.log(df['Adj Close']).diff().dropna()) - 1.0
    returns.name = name
    new_index = pd.to_datetime(returns.index) + pd.offsets.MonthBegin(-1)
    returns.index = new_index

    return returns
