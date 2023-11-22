import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np
from scipy.optimize import minimize

class LinearRegressionSM:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None

    def fit(self):
        X = sm.add_constant(self.right_hand_side)
        model = sm.OLS(self.left_hand_side, X).fit()
        self._model = model

    def get_params(self):
        beta_coefficients = self._model.params
        return pd.Series(beta_coefficients, name='Beta coefficients')

    def get_pvalues(self):
        p_values = self._model.pvalues
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, constraints):
        wald_test = self._model.wald_test(constraints)
        f_value = float(wald_test.statistic)
        p_value = float(wald_test.pvalue)
        return f'F-value: {f_value:.3}, p-value: {p_value:.3}'

    def get_model_goodness_values(self):
        adjusted_r_squared = self._model.rsquared_adj
        aic = self._model.aic
        bic = self._model.bic
        return f'Adjusted R-squared: {adjusted_r_squared:.3}, Akaike IC: {aic:.3}, Bayes IC: {bic:.3}'


class LinearRegressionNP:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.coefficients = None
        self.residuals = None
        self.p_values = None
        self.crs = None
        self.ars = None

    def fit(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

        n, k = X.shape
        self.residuals = y - X @ self.coefficients
        sigma_squared = np.sum(self.residuals ** 2) / (n - k)
        var_beta = np.linalg.inv(X.T @ X) * sigma_squared
        t_statistic = self.coefficients / np.sqrt(np.diag(var_beta))
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=n-k))

        rss = np.sum(self.residuals ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)

        self.crs = 1 - rss / tss
        self.ars = 1 - (rss / (n - k)) / (tss / (n - 1))
    def get_params(self):
        return pd.Series(self.coefficients, name='Beta coefficients')

    def get_pvalues(self):
        return pd.Series(self.p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, constraints):
        constraints = np.array(constraints)
        n = len(self.left_hand_side)
        m, k = constraints.shape
        sigma_squared = np.sum(self.residuals ** 2) / (n - k)
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))

        H = constraints @ np.linalg.inv(X.T @ X) @ constraints.T
        wald = (constraints @ self.coefficients).T @ np.linalg.inv(H) @ (constraints @ self.coefficients)
        wald = wald / m / sigma_squared
        p_value = 1 - stats.f.cdf(wald, dfn=m, dfd=n-k)

        return f'Wald: {wald:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        crs = self.crs
        ars = self.ars

        return f'Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}'


class LinearRegressionGLS:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.coefficients = None
        self.residuals = None
        self.p_values = None
        self.crs = None
        self.ars = None
        self.V_inv = None

    def fit(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        coef_as = np.linalg.inv(X.T @ X) @ X.T @ y

        res_as = y - X @ coef_as
        log_res_sq_as = np.log(res_as**2)

        coef_as2 = np.linalg.inv(X.T @ X) @ X.T @ log_res_sq_as

        self.V_inv = np.diag(1 / np.sqrt(np.exp(X @ coef_as2)))

        self.coefficients = np.linalg.inv(X.T @ self.V_inv @ X) @ (X.T @ self.V_inv @ y)

        n, k = X.shape
        self.residuals = y - X @ self.coefficients
        sigma_squared = np.sum(self.residuals ** 2) / (n - k)
        var_beta = np.linalg.inv(X.T @ self.V_inv @ X) * sigma_squared
        t_statistic = self.coefficients / np.sqrt(np.diag(var_beta))
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=n-k))

        rss = y.T @ self.V_inv @ X @ np.linalg.inv(X.T @ self.V_inv @ X) @ X.T @ self.V_inv @ y
        tss = y.T @ self.V_inv @ y

        self.crs = 1 - rss / tss
        self.ars = 1 - (rss / (n - k)) / (tss / (n - 1))
    def get_params(self):
        return pd.Series(self.coefficients, name='Beta coefficients')

    def get_pvalues(self):
        return pd.Series(self.p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, constraints):
        constraints = np.array(constraints)
        n = len(self.left_hand_side)
        m, k = constraints.shape
        sigma_squared = np.sum(self.residuals ** 2) / (n - k)
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))

        H = constraints @ np.linalg.inv(X.T @ self.V_inv @ X) @ constraints.T
        wald = (constraints @ self.coefficients).T @ np.linalg.inv(H) @ (constraints @ self.coefficients)
        wald = wald / m / sigma_squared
        p_value = 1 - stats.f.cdf(wald, dfn=m, dfd=n-k)

        return f'Wald: {wald:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        crs = self.crs
        ars = self.ars

        return f'Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}'


class LinearRegressionML:

    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.coefficients = None
        self.residuals = None
        self.p_values = None
        self.crs = None
        self.ars = None

    def calculate_neg_loglikelihood(self, params):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        beta0, beta1, beta2, beta3, sig = params
        betas = np.array([beta0, beta1, beta2, beta3])
        pred = X @ betas
        LL = np.sum(stats.norm.logpdf(y, pred, sig))

        return -LL

    def fit(self):
        opt = minimize(self.calculate_neg_loglikelihood, np.array([.1, .1, .1, .1, .1]), method='L-BFGS-B')
        beta0, beta1, beta2, beta3, sig = opt.x
        self.coefficients = np.array([beta0, beta1, beta2, beta3])

        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        n, k = X.shape
        self.residuals = y - X @ self.coefficients
        sigma_squared = np.sum(self.residuals ** 2) / (n - k)
        var_beta = np.linalg.inv(X.T @ X) * sigma_squared
        t_statistic = self.coefficients / np.sqrt(np.diag(var_beta))
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=n - k))

        rss = np.sum(self.residuals ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)

        self.crs = 1 - rss / tss
        self.ars = 1 - (rss / (n - k)) / (tss / (n - 1))

    def get_params(self):
        return pd.Series(self.coefficients, name='Beta coefficients')

    def get_pvalues(self):
        return pd.Series(self.p_values, name='P-values for the corresponding coefficients')

    def get_model_goodness_values(self):
        crs = self.crs
        ars = self.ars

        return f'Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}'