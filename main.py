import pandas as pd
from scipy import stats


def test_independence(df, var1, var2, condition_vars=None):
    """
    Test for the independence condition (var1 _|_ var2 | condition_vars) in df.

    Parameters
    ----------
    df: pandas Dataframe
        The dataset on which to test the independence condition.

    var1: str
        First variable in the independence condition.

    var2: str
        Second variable in the independence condition

    condition_vars: list
        List of variable names in given variables.

    Returns
    -------
    chi_stat: float
        The chi-square statistic for the test.

    p_value: float
        The p-value of the test

    dof: int
        Degrees of Freedom

    Examples
    --------
    >>> df = pd.read_csv('adult.csv')
    >>> chi, p, dof = test_independence(df, var1='Age', var2='Immigrant')
    >>> print("chi =", chi, "\np =", p, "\ndof =",dof)
    chi = 57.7514122288
    p = 8.60514815766e-12
    dof = 4
    >>> chi, p, dof = test_independence(df, var1='Education', var2='HoursPerWeek',
    ...                                 condition_vars=['Age', 'Immigrant', 'Sex'])
    >>> print("chi=", chi, "\np=", p, "\ndof=",dof)
    chi = 1360.65856663 
    p = 0.0 
    dof = 171
    """
    if not condition_vars:
        observed = pd.crosstab(df[var1], df[var2])
        chi_stat, p_value, dof, expected = stats.chi2_contingency(observed)

    else:
        observed_combinations = df.groupby(condition_vars).size().reset_index()
        chi_stat = 0
        dof = 0
        for combination in range(len(observed_combinations)):
            df_conditioned = df.copy()
            for condition_var in condition_vars:
                df_conditioned = df_conditioned.loc[df_conditioned.loc[:, condition_var] == observed_combinations.loc[combination, condition_var]]
            observed = pd.crosstab(df_conditioned[var1], df_conditioned[var2])
            chi, _, freedom, _ = stats.chi2_contingency(observed)
            chi_stat += chi
            dof += freedom
        p_value = 1.0 - stats.chi2.cdf(x=chi_stat, df=dof)

    return chi_stat, p_value, dof

import pgmpy

df = pd.read_csv("processed.cleveland.data", header=None)
df.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

linear_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]

print(df[linear_columns].mean())

df[linear_columns] = df[linear_columns] > df[linear_columns].mean()

print(df)

tests = [('cp', 'ca', ['num']), ('cp', 'exang', ['num']), ('trestbps', 'age', ['num', 'fbs', 'chol'])]

# hoge p-waarde independent, lage p-waarde dependent

for var1, var2, cond in tests:
    chi, p, dof = test_independence(df=df, var1=var1, var2=var2, condition_vars=cond)
    print(p)
    print(var1, 'and', var2, 'are', '' if p < 0.05 else 'not', 'dependent given', cond)
