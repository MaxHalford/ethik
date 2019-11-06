import collections
import functools
import itertools
import warnings

import colorlover as cl
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy import special
from tqdm import tqdm

from .utils import decimal_range, join_with_overlap, to_pandas, yield_masks
from .warnings import ConvergenceWarning

__all__ = ["BaseExplainer"]


def compute_ksis(x, target_means, max_iterations, tol):
    """Find good ksis for a variable and given target means.

    Args:
        x (pd.Series): The variable's values.
        target_means (iterator of floats): The means to reach by weighting the
            feature's values.
        max_iterations (int): The maximum number of iterations of gradient descent.
        tol (float): Stopping criterion. The gradient descent will stop if the mean absolute error
            between the obtained mean and the target mean is lower than tol, even if the maximum
            number of iterations has not been reached.

    Returns:
        dict: The keys are couples `(name of the variable, target mean)` and the
            values are the ksis. For instance:

                {
                    ("age", 20): 1.5,
                    ("age", 21): 1.6,
                    ...
                }
    """

    mean = x.mean()
    ksis = {}

    for target_mean in target_means:

        ksi = 0
        current_mean = mean
        n_iterations = 0

        while n_iterations < max_iterations:
            n_iterations += 1

            # Stop if the target mean has been reached
            if current_mean == target_mean:
                break

            # Update the sample weights and obtain the new mean of the distribution
            # TODO: if ksi * x is too large then sample_weights might only contain zeros, which
            # leads to hess being equal to 0
            lambdas = special.softmax(ksi * x)
            current_mean = np.average(x, weights=lambdas)

            # Do a Newton step using the difference between the mean and the
            # target mean
            grad = current_mean - target_mean
            hess = np.average((x - current_mean) ** 2, weights=lambdas)

            # We use a magic number for the step size if the hessian is nil
            step = (1e-5 * grad) if hess == 0 else (grad / hess)
            ksi -= step

            # Stop if the gradient is small enough
            if abs(grad) < tol:
                break

        # Warn the user if the algorithm didn't converge
        else:
            warnings.warn(
                message=(
                    f"Gradient descent failed to converge after {max_iterations} iterations "
                    + f"(name={x.name}, mean={mean}, target_mean={target_mean}, "
                    + f"current_mean={current_mean}, grad={grad}, hess={hess}, step={step}, ksi={ksi})"
                ),
                category=ConvergenceWarning,
            )

        ksis[(x.name, target_mean)] = ksi

    return ksis


class BaseExplainer:
    CAT_COL_SEP = " = "

    def __init__(
        self,
        alpha=0.05,
        n_taus=41,
        n_samples=1,
        sample_frac=0.8,
        conf_level=0.05,
        max_iterations=15,
        tol=1e-3,
        n_jobs=1,  # Parallelism is only worth it if the dataset is "large"
        verbose=True,
    ):
        if not 0 <= alpha < 0.5:
            raise ValueError(f"alpha must be between 0 and 0.5, got {alpha}")

        if not n_taus > 0:
            raise ValueError(
                f"n_taus must be a strictly positive integer, got {n_taus}"
            )

        if n_samples < 1:
            raise ValueError(f"n_samples must be strictly positive, got {n_samples}")

        if not 0 < sample_frac < 1:
            raise ValueError(f"sample_frac must be between 0 and 1, got {sample_frac}")

        if not 0 < conf_level < 0.5:
            raise ValueError(f"conf_level must be between 0 and 0.5, got {conf_level}")

        if not max_iterations > 0:
            raise ValueError(
                "max_iterations must be a strictly positive "
                f"integer, got {max_iterations}"
            )

        if not tol > 0:
            raise ValueError(f"tol must be a strictly positive number, got {tol}")

        self.alpha = alpha
        self.n_taus = n_taus
        self.n_samples = n_samples
        self.sample_frac = sample_frac if n_samples > 1 else 1
        self.conf_level = conf_level
        self.max_iterations = max_iterations
        self.tol = tol
        self.n_jobs = n_jobs
        self.verbose = verbose

    def get_metric_name(self, metric):
        """Get the name of the column in explainer's info dataframe to store the
        performance with respect of the given metric.

        Args:
            metric (callable): The metric to compute the model's performance.

        Returns:
            str: The name of the column.
        """
        name = metric.__name__
        if name in ["feature", "target", "label", "influence", "ksi"]:
            raise ValueError(
                f"Cannot use {name} as a metric name, already a column name"
            )
        return name

    def _fill_ksis(self, X_test, query):
        """

        Parameters:
            X_test (pd.DataFrame): A dataframe with categorical features ALREADY 
                one-hot encoded.
            query (pd.DataFrame):
        """
        if "ksi" not in query.columns:
            query["ksi"] = None

        X_test = pd.DataFrame(to_pandas(X_test))
        query_to_complete = query[query["ksi"].isnull()]
        ksis = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(compute_ksis)(
                x=X_test[feature],
                target_means=part["target"].unique(),
                max_iterations=self.max_iterations,
                tol=self.tol,
            )
            for feature, part in query_to_complete.groupby("feature")
        )
        ksis = dict(collections.ChainMap(*ksis))

        query["ksi"] = query.apply(
            lambda r: ksis.get((r["feature"], r["target"]), r["ksi"]), axis="columns"
        )
        query["ksi"] = query["ksi"].fillna(0.0)
        return query

    def _explain(
        self, X_test, y_pred, dest_col, key_cols, compute, query, compute_kwargs=None
    ):
        if compute_kwargs is None:
            compute_kwargs = {}
        if dest_col not in query.columns:
            query[dest_col] = None
            query[f"{dest_col}_low"] = None
            query[f"{dest_col}_high"] = None

        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))
        # One-hot encode the categorical features
        X_test = pd.get_dummies(data=X_test, prefix_sep=self.CAT_COL_SEP)

        if len(X_test) != len(y_pred):
            raise ValueError("X_test and y_pred are not of the same length")

        query = self._fill_ksis(X_test, query)

        query_to_complete = query[
            query["feature"].isin(X_test.columns)
            & query["label"].isin(y_pred.columns)
            & query[dest_col].isnull()
        ]

        if query_to_complete.empty:
            return query

        # `compute()` will return something like:
        # [
        #   [ # First batch
        #     (*key_cols1, sample_index1, computed_value1),
        #     (*key_cols2, sample_index2, computed_value2),
        #     ...
        #   ],
        #   ...
        # ]
        explanation = compute(
            X_test=X_test, y_pred=y_pred, query=query_to_complete, **compute_kwargs
        )
        # We group by the key to gather the samples and compute the confidence
        #  interval
        explanation = explanation.groupby(key_cols)[dest_col].agg(
            [
                # Mean influence
                (dest_col, "mean"),
                # Lower bound on the mean influence
                (f"{dest_col}_low", functools.partial(np.quantile, q=self.conf_level)),
                # Upper bound on the mean influence
                (
                    f"{dest_col}_high",
                    functools.partial(np.quantile, q=1 - self.conf_level),
                ),
            ]
        )

        # Merge the new queryrmation with the current queryrmation
        query = join_with_overlap(left=query, right=explanation, on=key_cols)
        return query

    def _compute_influence(self, X_test, y_pred, query):
        keys = query.groupby(["feature", "label", "ksi"]).groups.keys()
        return pd.DataFrame(
            [
                (
                    feature,
                    label,
                    ksi,
                    sample_index,
                    np.average(
                        y_pred[label][mask],
                        weights=special.softmax(ksi * X_test[feature][mask]),
                    ),
                )
                for (sample_index, mask), (feature, label, ksi) in tqdm(
                    itertools.product(
                        enumerate(
                            yield_masks(
                                n_masks=self.n_samples,
                                n=len(X_test),
                                p=self.sample_frac,
                            )
                        ),
                        keys,
                    ),
                    disable=not self.verbose,
                    total=len(keys) * self.n_samples,
                )
            ],
            columns=["feature", "label", "ksi", "sample_index", "influence"],
        )

    def _explain_influence(self, X_test, y_pred, query):
        return self._explain(
            X_test=X_test,
            y_pred=y_pred,
            dest_col="influence",
            key_cols=["feature", "label", "ksi"],
            compute=self._compute_influence,
            query=query,
        )

    def _compute_performance(self, X_test, y_pred, query, y_test, metric):
        metric_name = self.get_metric_name(metric)
        keys = query.groupby(["feature", "ksi"]).groups.keys()
        return pd.DataFrame(
            [
                (
                    feature,
                    ksi,
                    sample_index,
                    metric(
                        y_test[mask],
                        y_pred[mask],
                        sample_weight=special.softmax(ksi * X_test[feature][mask]),
                    ),
                )
                for (sample_index, mask), (feature, ksi) in tqdm(
                    itertools.product(
                        enumerate(
                            yield_masks(
                                n_masks=self.n_samples,
                                n=len(X_test),
                                p=self.sample_frac,
                            )
                        ),
                        keys,
                    ),
                    disable=not self.verbose,
                    total=len(keys) * self.n_samples,
                )
            ],
            columns=["feature", "ksi", "sample_index", metric_name],
        )

    def _explain_performance(self, X_test, y_test, y_pred, metric, query):
        metric_name = self.get_metric_name(metric)
        return self._explain(
            X_test=X_test,
            y_pred=y_pred,
            dest_col=metric_name,
            key_cols=["feature", "ksi"],
            compute=self._compute_performance,
            query=query,
            compute_kwargs=dict(y_test=np.asarray(y_test), metric=metric),
        )

    def _plot_explanation(
        self, explanation, col, y_label, colors=None, yrange=None, size=None
    ):
        features = explanation["feature"].unique()

        if colors is None:
            colors = {}
        elif type(colors) is str:
            colors = {feat: colors for feat in features}

        width = height = None
        if size is not None:
            width, height = size

        #  There are multiple features, we plot them together with taus
        if len(features) > 1:
            fig = go.Figure()

            for i, feat in enumerate(features):
                taus = explanation.query(f'feature == "{feat}"')["tau"]
                targets = explanation.query(f'feature == "{feat}"')["target"]
                y = explanation.query(f'feature == "{feat}"')[col]
                fig.add_trace(
                    go.Scatter(
                        x=taus,
                        y=y,
                        mode="lines+markers",
                        hoverinfo="y",
                        name=feat,
                        customdata=list(zip(taus, targets)),
                        marker=dict(color=colors.get(feat)),
                    )
                )

            fig.update_layout(
                margin=dict(t=50, r=50),
                xaxis=dict(
                    title="tau",
                    nticks=5,
                    showline=True,
                    showgrid=True,
                    zeroline=False,
                    linecolor="black",
                    gridcolor="#eee",
                ),
                yaxis=dict(
                    title=y_label,
                    range=yrange,
                    showline=True,
                    showgrid=True,
                    linecolor="black",
                    gridcolor="#eee",
                ),
                plot_bgcolor="white",
                width=width,
                height=height,
            )
            return fig

        #  There is only one feature, we plot it with its nominal values.
        feat = features[0]
        fig = go.Figure()
        x = explanation.query(f'feature == "{feat}"')["target"]
        y = explanation.query(f'feature == "{feat}"')[col]
        mean_row = explanation.query(f'feature == "{feat}" and tau == 0').iloc[0]

        if self.n_samples > 1:
            low = explanation.query(f'feature == "{feat}"')[f"{col}_low"]
            high = explanation.query(f'feature == "{feat}"')[f"{col}_high"]
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate((x, x[::-1])),
                    y=np.concatenate((low, high[::-1])),
                    name=f"{self.conf_level * 100}% - {(1 - self.conf_level) * 100}%",
                    fill="toself",
                    fillcolor=colors.get(feat),
                    line_color="rgba(0, 0, 0, 0)",
                    opacity=0.3,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                hoverinfo="x+y",
                showlegend=False,
                marker=dict(color=colors.get(feat)),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[mean_row["target"]],
                y=[mean_row[col]],
                text=["Dataset mean"],
                showlegend=False,
                mode="markers",
                name="Original mean",
                hoverinfo="text",
                marker=dict(symbol="x", size=9, color=colors.get(feat)),
            )
        )
        fig.update_layout(
            margin=dict(t=50, r=50),
            xaxis=dict(title=f"Average {feat}", zeroline=False),
            yaxis=dict(title=y_label, range=yrange, showline=True),
            plot_bgcolor="white",
            width=width,
            height=height,
        )
        fig.update_xaxes(
            showline=True, showgrid=True, linecolor="black", gridcolor="#eee"
        )
        fig.update_yaxes(
            showline=True, showgrid=True, linecolor="black", gridcolor="#eee"
        )
        return fig

    def compute_distributions(self, X_test, targets, bins, y_pred=None, density=True):
        query = pd.DataFrame(
            dict(
                feature=[X_test.name] * len(targets),
                target=targets,
                label=[""] * len(targets),
            )
        )
        ksis = self._fill_ksis(X_test, query)["ksi"]

        distributions = {}
        for ksi, target in zip(ksis, targets):
            weights = special.softmax(ksi * X_test)
            densities, edges = np.histogram(
                y_pred if y_pred is not None else X_test,
                bins=bins,
                weights=weights,
                density=density,
            )
            distributions[target] = (
                edges,
                densities,
                np.average(y_pred, weights=weights) if y_pred is not None else None,
            )
        return distributions

    def plot_distributions(
        self,
        X_test,
        y_pred=None,
        bins=10,
        targets=None,
        colors=None,
        dataset_color="black",
        size=None,
    ):
        if targets is None:
            targets = []

        if colors is None:
            colors = cl.interp(
                cl.scales["11"]["qual"]["Paired"],
                len(targets) + 1,  #  +1 otherwise it raises an error if ksis is empty
            )

        if dataset_color is not None:
            targets = [X_test.mean(), *targets]  # Add the original mean
            colors = [dataset_color, *colors]

        fig = go.Figure()
        shapes = []
        distributions = self.compute_distributions(
            X_test=X_test, targets=targets, bins=bins, y_pred=y_pred
        )

        for i, (target, color) in enumerate(zip(targets, colors)):
            mean = target
            trace_name = f"E[{X_test.name}] = {mean:.2f}"
            edges, densities, y_pred_mean = distributions[target]
            if y_pred is not None:
                mean = y_pred_mean
                trace_name += f", E[{y_pred.name}] = {mean:.2f}"
            if i == 0:
                trace_name += " (dataset)"

            fig.add_bar(
                x=edges[:-1],
                y=densities,
                name=trace_name,
                opacity=0.5,
                marker=dict(color=color),
            )
            shapes.append(
                go.layout.Shape(
                    type="line",
                    x0=mean,
                    y0=0,
                    x1=mean,
                    y1=1,
                    yref="paper",
                    line=dict(color=color, width=1),
                )
            )

        width = height = None
        if size is not None:
            width, height = size

        fig.update_layout(
            bargap=0,
            barmode="overlay",
            showlegend=True,
            xaxis=dict(
                title=y_pred.name if y_pred is not None else X_test.name,
                linecolor="black",
            ),
            yaxis=dict(title="Probability density", linecolor="black"),
            shapes=shapes,
            width=width,
            height=height,
            plot_bgcolor="#FFF",
        )
        return fig
