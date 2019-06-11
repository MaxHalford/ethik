import base64
import decimal
import itertools
import os
import random
import string

from IPython import display
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


__all__ = ['Explainer']


def decimal_range(start: float, stop: float, step: float):
    """Like the `range` function but works for decimal values.

    This is more accurate than using `np.arange` because it doesn't introduce
    any round-off errors.

    """
    start = decimal.Decimal(str(start))
    stop = decimal.Decimal(str(stop))
    step = decimal.Decimal(str(step))
    while start <= stop:
        yield float(start)
        start += step


def random_id(size=20, chars=string.ascii_uppercase + string.digits):
    """Returns a random identifier.

    This is simply used for identifying a <div> when rendering HTML.

    """
    return 'i' + ''.join(random.choice(chars) for _ in range(size))


def to_pandas(x):
    """Converts an array-like to a Series or a DataFrame depending on the dimensionality."""

    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x

    if isinstance(x, np.ndarray):
        if x.ndim > 2:
            raise ValueError('x must have 1 or 2 dimensions')
        if x.ndim == 2:
            return pd.DataFrame(x)
        return pd.Series(x)

    return to_pandas(np.asarray(x))


def compute_lambdas(x, epsilons, max_iterations=5):
    """Finds a good lambda for a variable and a given epsilon value.

    Parameters:
        x (array-like)
        eps (float)

    """

    # from scipy import optimize

    # new_mean = x.mean() + eps

    # res = optimize.minimize(
    #     fun=lambda λ: np.log(np.mean(np.exp(λ * x))) - λ * new_mean,
    #     options={'maxiter': 10},
    #     x0=0.,
    #     method='Nelder-Mead'
    # )

    # return res.x[0]

    mean = x.mean()
    lambdas = np.empty(len(epsilons))

    for i, eps in enumerate(epsilons):

        λ = 0
        current_mean = mean
        target_mean = mean + eps

        for _ in range(max_iterations):

            # Update the sample weights and see where the mean is
            sample_weights = np.exp(λ * x)
            sample_weights = sample_weights / sum(sample_weights)
            current_mean = np.average(x, weights=sample_weights)

            # Do a Newton step using the difference between the mean and the
            # target mean
            grad = current_mean - target_mean
            hess = np.average((x - current_mean) ** 2, weights=sample_weights)
            step = grad / hess
            λ -= step

        lambdas[i] = λ

    return lambdas


def compute_pred_means(X, lambdas, y_pred):
    return pd.DataFrame(
        data={
            col: [
                np.average(y_pred, weights=np.exp(λ * X[col]))
                for tau, λ in lambdas[col].items()
            ]
            for col in X.columns
        }
    )


class Explainer():
    """Explains the bias and reliability of model predictions.

    Parameters:
        alpha (float): A `float` between `0` and `0.5` which indicates by how close the `Explainer`
            should look at extreme values of a distribution. The closer to zero, the more so
            extreme values will be accounted for. The default is `0.05` which means that all values
            beyond the 5th and 95th quantiles are ignored.
        n_taus (int): The number of τ values to consider. The results will be more fine-grained the
            higher this value is. However the computation time increases linearly with `n_taus`.
            The default is `41` and corresponds to each τ being separated by it's neighbors by
            `0.05`.
        max_iterations (int): The maximum number of iterations used when applying the Newton step
            of the optimization procedure.

    """

    def __init__(self, alpha=0.05, n_taus=41, max_iterations=5, n_jobs=-1, verbose=False):
        self.alpha = alpha
        self.n_taus = n_taus
        self.max_iterations = max_iterations
        self.n_jobs = n_jobs
        self.verbose = verbose

    def check_parameters(self):
        """Raises a `ValueError` if any parameter is invalid."""
        if not 0 < self.alpha < 0.5:
            raise ValueError('alpha must be between 0 and 0.5, got '
                             f'{self.alpha}')

        if not self.n_taus > 0:
            raise ValueError('n_taus must be a strictly positive integer, got '
                             f'{self.n_taus}')

        if not self.max_iterations > 0:
            raise ValueError('max_iterations must be a strictly positive '
                             f'integer, got {self.max_iterations}')

    def make_epsilons(self, X, taus):
        """Returns a DataFrame containing espilons for each (column, tau) pair.

        Attributes:
            X (pandas.DataFrame)
            taus (list of floats)

        Returns:
            pandas.DataFrame

        """

        q_mins = X.quantile(q=self.alpha).to_dict()
        q_maxs = X.quantile(q=1 - self.alpha).to_dict()
        means = X.mean().to_dict()

        return pd.DataFrame({
            col: {
                tau: tau * (
                    (means[col] - q_mins[col]) if tau < 0
                    else (q_maxs[col] - means[col])
                )
                for tau in taus
            }
            for col in X.columns
        })

    def fit(self, X):
        """Fits the explainer to a tabular dataset.

        During a `fit` call, the following steps are taken:

        1. A list of $\tau$ values is generated using `n_taus`. The $\tau$ values range from -1 to 1.
        2. A grid of $\eps$ values is generated for each $\tau$ and for each variable. Each $\eps$ represents a shift from a variable's mean towards a particular quantile.
        3. A grid of $\lambda$ values is generated for each $\eps$. Each $\lambda$ corresponds to the optimal parameter that has to be used to weight the observations in order for the average to reach the associated $\eps$ shift.

        Parameters:
            X (`pandas.DataFrame` or `numpy.ndarray`)

        """

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Check the parameters are correct
        self.check_parameters()

        # Make the taus
        tau_precision = 2 / (self.n_taus - 1)
        self.taus = list(decimal_range(-1, 1, tau_precision))

        # Make the epsilons
        X_num = X.select_dtypes(exclude=['object', 'category'])
        self.epsilons = self.make_epsilons(X_num, self.taus)

        # Find a lambda for each (column, quantile) pair
        lambdas = joblib.Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )(
            joblib.delayed(compute_lambdas)(
                x=X[col],
                epsilons=self.epsilons[col].to_numpy(),
                max_iterations=self.max_iterations
            )
            for col in X_num.columns
        )
        self.lambdas = pd.DataFrame(data=dict(zip(X_num.columns, lambdas)), index=self.taus)

        return self

    def nominal_values(self, X):
        return pd.DataFrame({
            col: X[col].mean() + self.epsilons[col]
            for col in X.columns
        }, index=pd.Index(self.taus, name=r'$\tau$'))

    def explain_predictions(self, X, y_pred):
        """Returns a DataFrame containing average predictions for each (column, tau) pair.

        """

        # Coerce X and y_pred to DataFrames
        X = pd.DataFrame(to_pandas(X))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        # Compute the average predictions for each (column, tau) pair per label
        preds = joblib.Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )(
            joblib.delayed(compute_pred_means)(
                X=X,
                lambdas=self.lambdas,
                y_pred=y_pred[label]
            )
            for label in y_pred.columns
        )
        preds = pd.concat(preds, axis='columns', ignore_index=True)
        preds.columns = pd.MultiIndex.from_tuples(
            tuples=itertools.product(y_pred.columns, X.columns),
            names=['labels', 'features']
        )

        preds.index = self.taus
        preds.index.name = r'$\tau$'

        # Remove unnecessary column levels
        if len(preds.columns.unique('labels')) == 1:
            preds.columns = preds.columns.droplevel('labels')
        if isinstance(preds.columns, pd.MultiIndex) and len(preds.columns.unique('features')) == 1:
            preds.columns = preds.columns.droplevel('features')

        return preds

    def plot_predictions(self, X, y_pred, ax=None):
        """Plots predicted means against variables values.

        If a single column is provided then the x-axis is made of the nominal
        values from that column. However if a list of columns is passed then
        the x-axis contains the tau values. This method makes use of the
        `explain_predictions` method.

        """
        means = self.explain_predictions(X=X, y_pred=y_pred)

        # Create a plot if none has been provided
        ax = plt.axes() if ax is None else ax

        def to_label(col):
            if isinstance(col, tuple):
                return f'{col[0]} - {col[1]}'
            return col

        if len(means.columns) == 1:
            col = means.columns[0]
            x = self.nominal_values(X)[col]
            ax.plot(x, means[col].values, label=to_label(col))
        else:
            for col in means.columns:
                ax.plot(means[col], label=to_label(col))

        # Add the legend if necessary
        if len(means.columns) > 1:
            ax.legend()

        # Set the x-axis label appropriately
        if len(means.columns) == 1:
            ax.set_xlabel(means.columns[0])
        else:
            ax.set_xlabel(r'$\tau$')

        # Prettify the plot
        ax.grid(True)
        ax.set_ylabel('Target mean')
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)

        return ax

    def explain_metric(self, X, y, y_pred, metric):
        """Returns a DataFrame with metric values for each (column, tau) pair.

        Parameters:
            metric (callable): A function that evaluates the quality of a set of predictions. Must
                have the following signature: `metric(y_true, y_pred, sample_weights)`. Most
                metrics from scikit-learn will work.

        """

        X = pd.DataFrame(to_pandas(X))

        metrics = pd.DataFrame(
            data={
                col: [
                    metric(y, y_pred, sample_weight=np.exp(λ * X[col]))
                    for tau, λ in self.lambdas[col].items()
                ]
                for col in X.columns
            },
            index=self.taus
        )

        # If there a single column we can index with epsilons instead of taus
        if isinstance(metrics, pd.Series):
            mean = X[metrics.name].mean()
            metrics.index = [mean + eps for eps in self.epsilons[metrics.name]]

        if len(metrics.columns) == 1:
            return metrics[metrics.columns[0]]
        return metrics

    def plot_metric(self, X, y, y_pred, metric, ax=None):
        """Plots metric values against variable values.

        If a single column is provided then the x-axis is made of the nominal
        values from that column. However if a list of columns is passed then
        the x-axis contains the tau values. This method makes use of the
        `explain_metric` method.

        """

        metrics = self.explain_metric(X=X, y=y, y_pred=y_pred, metric=metric)

        # Create a plot if none is provided
        ax = plt.axes() if ax is None else ax

        if isinstance(metrics, pd.Series):

            # If a single column is provided then we can plot scores against
            # values
            ax.plot(metrics)
            ax.set_xlabel(metrics.name)

        else:

            # If more than column is provided then we can plot scores against
            # values for each column OR plot scores against taus on one single
            # axis
            for col in metrics.columns:
                ax.plot(metrics[col], label=col)
            ax.legend()
            ax.set_xlabel(r'$\tau$')

        # Prettify the plot
        ax.grid(True)
        ax.set_ylabel(metric.__name__)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)

        return ax

    def initjs(self):

        # Load JavaScript dependencies
        display.display(display.Javascript('''
            require.config({
                paths: {
                    d3: "https://d3js.org/d3.v5.min"
                }
            });
        '''))

        # Load the JavaScript logo
        here = os.path.dirname(__file__)
        logo_path = os.path.join(here, 'resources', 'js_logo.png')
        with open(logo_path, 'rb') as f:
            logo_data = f.read()

        display.display(display.HTML(string.Template('''
            <div align="center">
                <img src="data:image/png;base64,$logo" />
            </div>
        ''').substitute(logo=base64.b64encode(logo_data).decode('utf-8'))))
