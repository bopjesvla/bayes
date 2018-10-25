import pandas as pd
import numpy as np
from scipy import stats
import re

def load_independencies(fn):
    text = open(fn).read()
    lists = [re.findall(r'Age|Sex|Diabetes|Thal|(?<=\()[a-z]+(?=\))', t) for t in text.split('\n')
             if t != '']
    print(lists)
    lists = [[e.lower() for e in l] for l in lists]
    tests = [(l[0], l[1], l[2:]) for l in lists if 'Diabetes' not in l]
    return lists

tests = load_independencies('origineel.txt')

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

print(df[linear_columns].median())

df[linear_columns] = df[linear_columns] > df[linear_columns].median()

print(df)

tests = [
# ('cp', 'exang', ['num', 'diabetes']),
# ('cp', 'fbs', ['num', 'diabetes']),
('cp', 'ca', ['num']),
('cp', 'thalach', ['num']),
('cp', 'restecg', ['num']),
('cp', 'oldpeak', ['num']),
# ('cp', 'chol', ['num', 'diabetes']),
('cp', 'slope', ['num']),
('cp', 'trestbps', ['num', 'fbs', 'chol']),
# ('cp', 'trestbps', ['num', 'diabetes']),
    # ('cp', 'age', ['num', 'diabetes']),
    # ('cp', 'sex', ['num', 'diabetes']),
('cp', 'thal', ['num']),
# ('exang', 'fbs', ['num', 'diabetes']),
('exang', 'ca', ['num']),
('exang', 'thalach', ['num']),
('exang', 'restecg', ['num']),
('exang', 'oldpeak', ['num']),
# ('exang', 'chol', ['num', 'diabetes']),
('exang', 'slope', ['num']),
('exang', 'trestbps', ['num', 'fbs', 'chol']),
# ('exang', 'trestbps', ['num', 'diabetes']),
    # ('exang', 'age', ['num', 'diabetes']),
    # ('exang', 'sex', ['num', 'diabetes']),
('exang', 'thal', ['num']),
('fbs', 'ca', ['num']),
('fbs', 'thalach', ['num']),
('fbs', 'restecg', ['num']),
('fbs', 'oldpeak', ['num']),
('fbs', 'slope', ['num']),
('fbs', 'thal', ['num']),
('ca', 'thalach', ['num']),
('ca', 'restecg', ['num']),
('ca', 'oldpeak', ['num']),
('ca', 'chol', ['num']),
('ca', 'slope', ['num']),
('ca', 'trestbps', ['num']),
('ca', 'age', ['num']),
    # ('ca', 'diabetes', ['num']),
('ca', 'sex', ['num']),
('ca', 'thal', ['num']),
('thalach', 'restecg', ['num']),
('thalach', 'oldpeak', ['num']),
('thalach', 'chol', ['num']),
('thalach', 'slope', ['num']),
('thalach', 'trestbps', ['num']),
('thalach', 'age', ['num']),
    # ('thalach', 'diabetes', ['num']),
('thalach', 'sex', ['num']),
('thalach', 'thal', ['num']),
('restecg', 'oldpeak', ['num']),
('restecg', 'chol', ['num']),
('restecg', 'slope', ['num']),
('restecg', 'trestbps', ['num']),
('restecg', 'age', ['num']),
    # ('restecg', 'diabetes', ['num']),
('restecg', 'sex', ['num']),
('restecg', 'thal', ['num']),
('oldpeak', 'chol', ['num']),
('oldpeak', 'slope', ['num']),
('oldpeak', 'trestbps', ['num']),
('oldpeak', 'age', ['num']),
    # ('oldpeak', 'diabetes', ['num']),
('oldpeak', 'sex', ['num']),
('oldpeak', 'thal', ['num']),
('chol', 'slope', ['num']),
    # ('chol', 'diabetes', ['fbs']),
('chol', 'thal', ['num']),
('slope', 'trestbps', ['num']),
('slope', 'age', ['num']),
    # ('slope', 'diabetes', ['num']),
('slope', 'sex', ['num']),
('slope', 'thal', ['num']),
('trestbps', 'age', ['num', 'fbs', 'chol']),
    # ('trestbps', 'diabetes', ['num', 'fbs', 'chol']),
('trestbps', 'sex', ['num', 'fbs', 'chol']),
('trestbps', 'thal', ['num']),
    # ('age', 'diabetes', ['fbs']),
('age', 'sex', []),
('age', 'thal', ['num']),
    # ('diabetes', 'sex', ['fbs']),
    # ('diabetes', 'thal', ['num']),
('sex', 'thal', ['num'])]


# hoge p-waarde independent, lage p-waarde dependent

for var1, var2, cond in tests:
    chi, p, dof = test_independence(df=df, var1=var1, var2=var2, condition_vars=cond)
    effect_size = np.sqrt(chi * chi / len(df) / dof)
    if p > 0.05:
        continue
    print(var1, 'and', var2, 'are', '' if p < 0.05 else 'not', 'dependent given', cond)
    print('effect size =', effect_size)
    print(p)

# cp and oldpeak are  dependent given ['num']
# effect size = 0.43626743727751777
# 0.011028759792580445
# exang and thalach are  dependent given ['num']
# effect size = 0.41649772602405216
# 0.006265886405995236
# fbs and ca are  dependent given ['num']
# effect size = 0.46312485408484716
# 0.010511655849716361
# ca and age are  dependent given ['num']
# effect size = 0.4817054697309406
# 0.007076949570464897
# thalach and slope are  dependent given ['num']
# effect size = 0.8276253063131302
# 1.7250533146384583e-06
# thalach and age are  dependent given ['num']
# effect size = 0.45558106965700795
# 0.0033009178368617054
# thalach and thal are  dependent given ['num']
# effect size = 0.35108995982893976
# 0.04793932464994066
# restecg and chol are  dependent given ['num']
# effect size = 0.35296507673161354
# 0.030478990520818372
# oldpeak and slope are  dependent given ['num']
# effect size = 1.151349936443901
# 8.276490603975617e-10
# sex and thal are  dependent given ['num']
# effect size = 0.5862403256722045
# 0.00041183761905361216
