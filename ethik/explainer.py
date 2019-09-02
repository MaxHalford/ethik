import collections
import decimal
import functools

import joblib
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go


__all__ = ["Explainer"]


def plot(fig, inline=False):
    if inline:
        plotly.offline.init_notebook_mode(connected=True)
        return plotly.offline.iplot(fig)
    return plotly.offline.plot(fig, auto_open=True)


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


def to_pandas(x):
    """Converts an array-like to a Series or a DataFrame depending on the dimensionality."""

    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x

    if isinstance(x, np.ndarray):
        if x.ndim > 2:
            raise ValueError("x must have 1 or 2 dimensions")
        if x.ndim == 2:
            return pd.DataFrame(x)
        return pd.Series(x)

    return to_pandas(np.asarray(x))


def merge_dicts(dicts):
    return dict(collections.ChainMap(*dicts))


def subsample(x, sample_size, seed, replace=False):
    if sample_size == 1:
        return x
    sample_size = int(sample_size * len(x))
    return np.random.RandomState(seed).choice(x, sample_size, replace=replace)


def compute_lambdas(x, target_means, seed, sample_size, max_iterations=5):
    """Finds a good lambda for a variable and a given epsilon value."""

    feature = x.name
    x = subsample(x, sample_size, seed)
    mean = x.mean()
    lambdas = {}

    for target_mean in target_means:

        λ = 0
        current_mean = mean

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

        lambdas[(feature, target_mean, seed)] = λ

    return lambdas


def compute_bias(y_pred, x, lambdas, seed, sample_size):
    feature = x.name
    label = y_pred.name
    x = subsample(x, sample_size, seed)
    y_pred = subsample(y_pred, sample_size, seed)
    return {
        (feature, label, λ, seed): np.average(y_pred, weights=np.exp(λ * x))
        for λ in lambdas
    }


def compute_performance(y_test, y_pred, metric, x, lambdas, seed, sample_size):
    feature = x.name
    x = subsample(x, sample_size, seed)
    y_pred = subsample(y_pred, sample_size, seed)
    y_test = subsample(y_test, sample_size, seed)
    return {
        (feature, λ, seed): metric(y_test, y_pred, sample_weight=np.exp(λ * x))
        for λ in lambdas
    }


def metric_to_col(metric):
    # TODO: what about lambda metrics?
    # TODO: use a prefix to avoid conflicts with other columns?
    return metric.__name__


def find_seeds(X_test, n_samples, sample_size, min_target_mean, max_target_mean):
    if n_samples == 1:
        yield 0  # Cannot use None or NaN as pandas.groupby() would skip them
        return

    n_seeds_valid = seed = 0
    # TODO: potential infinite loop?
    while n_seeds_valid < n_samples:
        x = subsample(X_test, sample_size, seed)
        if np.min(x) < min_target_mean and np.max(x) > max_target_mean:
            n_seeds_valid += 1
            yield seed
        seed += 1  # TODO: generate more unpredictable seeds?


def get_quantile_confint(series, level, axis=None):
    return (
        np.quantile(series, level, axis=axis),
        np.quantile(series, 1 - level, axis=axis),
    )


def check_conf_level(level):
    if level is None:
        return
    if not 0 < level < 0.5:
        raise ValueError(f"conf_level must be between 0 and 0.5. Got {level}")


class Explainer:
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

    def __init__(
        self,
        alpha=0.05,
        n_taus=41,
        max_iterations=5,
        n_jobs=-1,
        verbose=False,
        n_samples=1,
        sample_size=0.8,
    ):
        if not 0 < alpha < 0.5:
            raise ValueError("alpha must be between 0 and 0.5, got " f"{alpha}")

        if not n_taus > 0:
            raise ValueError(
                "n_taus must be a strictly positive integer, got " f"{n_taus}"
            )

        if not max_iterations > 0:
            raise ValueError(
                "max_iterations must be a strictly positive "
                f"integer, got {max_iterations}"
            )

        if n_samples < 1:
            raise ValueError(f"n_samples must be strictly positive, got {n_samples}")

        if not 0 < sample_size < 1:
            raise ValueError(f"sample_size must be between 0 and 1, got {sample_size}")

        #  TODO: one column per performance metric
        self.alpha = alpha
        self.n_taus = n_taus
        self.max_iterations = max_iterations
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.metric_cols = set()
        self.n_samples = n_samples
        self.sample_size = sample_size if n_samples > 1 else 1
        self.info = pd.DataFrame(
            columns=["seed", "feature", "tau", "value", "lambda", "label", "bias"]
        )

    @property
    def taus(self):
        tau_precision = 2 / (self.n_taus - 1)
        return list(decimal_range(-1, 1, tau_precision))

    @property
    def features(self):
        return self.info["feature"].unique().tolist()

    def _fit(self, X_test, y_pred):
        """Fits the explainer to a tabular dataset.

        During a `fit` call, the following steps are taken:

        1. A list of $\tau$ values is generated using `n_taus`. The $\tau$ values range from -1 to 1.
        2. A grid of $\eps$ values is generated for each $\tau$ and for each variable. Each $\eps$ represents a shift from a variable's mean towards a particular quantile.
        3. A grid of $\lambda$ values is generated for each $\eps$. Each $\lambda$ corresponds to the optimal parameter that has to be used to weight the observations in order for the average to reach the associated $\eps$ shift.

        Parameters:
            X_test (`pandas.DataFrame` or `numpy.ndarray`)
            y_pred (`pandas.DataFrame` or `numpy.ndarray`)

        """

        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        X_test = X_test[X_test.columns.difference(self.features)]
        if X_test.empty:
            return self.info

        # Make the epsilons
        q_mins = X_test.quantile(q=self.alpha).to_dict()
        q_maxs = X_test.quantile(q=1 - self.alpha).to_dict()
        means = X_test.mean().to_dict()
        X_test_num = X_test.select_dtypes(exclude=["object", "category"])
        additional_info = pd.concat(
            [
                pd.DataFrame(
                    {
                        "tau": self.taus,
                        "value": [
                            means[col]
                            + tau
                            * (
                                (means[col] - q_mins[col])
                                if tau < 0
                                else (q_maxs[col] - means[col])
                            )
                            for tau in self.taus
                        ],
                        "feature": col,
                        "label": y,
                        "seed": seed,
                    }
                )
                for col in X_test_num.columns
                for y in y_pred.columns
                for seed in find_seeds(
                    X_test_num[col],
                    self.n_samples,
                    self.sample_size,
                    q_mins[col],
                    q_maxs[col],
                )
            ],
            ignore_index=True,
        )
        self.info = self.info.append(additional_info, ignore_index=True, sort=False)

        # Find a lambda for each (column, espilon) pair
        lambdas = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            joblib.delayed(compute_lambdas)(
                x=X_test[col],
                target_means=part["value"].unique(),
                seed=seed,
                sample_size=self.sample_size,
                max_iterations=self.max_iterations,
            )
            for col, part in self.info.groupby("feature")
            for seed in part["seed"]
            if col in X_test
        )
        lambdas = merge_dicts(lambdas)
        self.info["lambda"] = self.info.apply(
            lambda r: lambdas.get((r["feature"], r["value"], r["seed"]), r["lambda"]),
            axis="columns",
        )

        return self.info

    def explain_bias(self, X_test, y_pred):
        """Returns a DataFrame containing average predictions for each (column, tau) pair.

        """
        # Coerce X_test and y_pred to DataFrames
        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        self._fit(X_test, y_pred)

        queried_features = X_test.columns.tolist()
        to_explain = self.info["feature"][self.info["bias"].isnull()].unique()
        X_test = X_test[X_test.columns.intersection(to_explain)]

        # Discard the features that are not relevant
        relevant = self.info.query(f"feature in {X_test.columns.tolist()}")

        # Compute the average predictions for each (column, tau) pair per label
        biases = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            joblib.delayed(compute_bias)(
                y_pred=y_pred[label],
                x=X_test[feat],
                lambdas=part["lambda"].unique(),
                seed=seed,
                sample_size=self.sample_size,
            )
            for label in y_pred.columns
            for (feat, seed), part in relevant.groupby(["feature", "seed"])
        )
        biases = merge_dicts(biases)
        self.info["bias"] = self.info.apply(
            lambda r: biases.get(
                (r["feature"], r["label"], r["lambda"], r["seed"]), r["bias"]
            ),
            axis="columns",
        )
        return self.info.query(f"feature in {queried_features}")

    def rank_by_bias(self, X_test, y_pred):
        """Returns a DataFrame containing the importance of each feature.

        """

        def get_importance(group):
            """Computes the average absolute difference in bias changes per tau increase."""
            baseline = group.query("tau == 0").iloc[0]["bias"]
            return (group["bias"] - baseline).abs().mean()

        return (
            self.explain_bias(X_test=X_test, y_pred=y_pred)
            .groupby(["label", "feature", "seed"])
            .apply(get_importance)
            .to_frame("importance")
            .reset_index()
        )

    def explain_performance(self, X_test, y_test, y_pred, metric):
        """Returns a DataFrame with metric values for each (column, tau) pair.

        Parameters:
            metric (callable): A function that evaluates the quality of a set of predictions. Must
                have the following signature: `metric(y_test, y_pred, sample_weights)`. Most
                metrics from scikit-learn will work.

        """
        if len(y_test) != len(y_pred):
            raise ValueError("y_test and y_pred must be of the same length")

        metric_col = metric_to_col(metric)
        if metric_col not in self.info.columns:
            self.info[metric_col] = None
        self.metric_cols.add(metric_col)

        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = to_pandas(y_pred)

        self._fit(X_test, y_pred)

        # Discard the features for which the score has already been computed
        queried_features = X_test.columns.tolist()
        to_explain = self.info["feature"][self.info[metric_col].isnull()].unique()
        X_test = X_test[X_test.columns.intersection(to_explain)]
        relevant = self.info.query(f"feature in {X_test.columns.tolist()}")

        # Compute the metric for each (feature, lambda) pair
        scores = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            joblib.delayed(compute_performance)(
                y_test=y_test,
                y_pred=y_pred,
                metric=metric,
                x=X_test[feat],
                lambdas=part["lambda"].unique(),
                seed=seed,
                sample_size=self.sample_size,
            )
            for (feat, seed), part in relevant.groupby(["feature", "seed"])
        )
        scores = merge_dicts(scores)
        self.info[metric_col] = self.info.apply(
            lambda r: scores.get((r["feature"], r["lambda"], r["seed"]), r[metric_col]),
            axis="columns",
        )

        return self.info.query(f"feature in {queried_features}")

    def rank_by_performance(self, X_test, y_test, y_pred, metric):
        metric_col = metric_to_col(metric)

        def get_aggregates(df):
            return pd.Series(
                [df[metric_col].min(), df[metric_col].max()], index=["min", "max"]
            )

        return (
            self.explain_performance(X_test, y_test, y_pred, metric)
            .groupby(["feature", "seed"])
            .apply(get_aggregates)
            .reset_index()
        )

    @classmethod
    def _make_explanation_fig(
        cls, explanation, y_col, y_label, with_taus=False, colors=None, conf_level=None
    ):
        check_conf_level(conf_level)
        if colors is None:
            colors = {}
        features = explanation["feature"].unique()

        if with_taus:
            fig = go.Figure()
            for feat in features:
                x = explanation.query(f"feature == '{feat}'")["tau"].unique()
                ys = [
                    part[y_col].values
                    for _, part in explanation.query(f'feature == "{feat}"').groupby(
                        "seed"
                    )
                ]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.mean(ys, axis=0),
                        mode="lines+markers",
                        hoverinfo="x+y+text",
                        name=feat,
                        text=[
                            f"{feat} = {val}"
                            for val in explanation.query(f'feature == "{feat}"')[
                                "value"
                            ]
                        ],
                        marker=dict(color=colors.get(feat)),
                    )
                )
            fig.update_layout(
                margin=dict(t=50, r=50),
                xaxis=dict(title="tau", zeroline=False),
                yaxis=dict(title=y_label, range=[0, 1], showline=True, tickformat="%"),
                plot_bgcolor="white",
            )
            return fig

        figures = {}
        for feat in features:
            fig = go.Figure()
            x = explanation.query(f'feature == "{feat}"')["value"].unique()
            ys = [
                part[y_col].values
                for _, part in explanation.query(f'feature == "{feat}"').groupby("seed")
            ]
            if conf_level is not None:
                low, high = get_quantile_confint(ys, conf_level, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate((x, x[::-1])),
                        y=np.concatenate((low, high[::-1])),
                        name=f"{conf_level * 100}% quantiles",
                        fill="toself",
                        fillcolor="#eee",  # TODO: same color as mean line?
                        line_color="rgba(0, 0, 0, 0)",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=np.mean(ys, axis=0),
                    name="Mean",
                    mode="lines+markers",
                    hoverinfo="x+y",
                    marker=dict(color=colors.get(feat)),
                )
            )
            fig.update_layout(
                margin=dict(t=50, r=50),
                xaxis=dict(title=f"Mean {feat}", zeroline=False),
                yaxis=dict(title=y_label, range=[0, 1], showline=True, tickformat="%"),
                plot_bgcolor="white",
            )
            figures[feat] = fig
        return figures

    @classmethod
    def make_bias_fig(cls, explanation, **kwargs):
        labels = explanation["label"].unique()
        y_label = f"Proportion of {labels[0]}"  #  Single class
        return cls._make_explanation_fig(
            explanation, y_col="bias", y_label=y_label, **kwargs
        )

    @classmethod
    def make_performance_fig(cls, explanation, metric, **kwargs):
        metric_col = metric_to_col(metric)
        return cls._make_explanation_fig(
            explanation, y_col=metric_col, y_label=metric_col, **kwargs
        )

    @classmethod
    def _make_ranking_fig(
        cls, ranking, score_column, title, colors=None, conf_level=None
    ):
        sorted_means = ranking.groupby("feature").mean().sort_values(by=[score_column])
        sorted_features = sorted_means.index
        low_errors = []
        high_errors = []

        if conf_level is not None:
            for i, feat in enumerate(sorted_features):
                scores = ranking.query(f"feature == '{feat}'")[score_column]
                low, high = get_quantile_confint(scores, conf_level)
                mean = sorted_means[score_column][feat]
                low_errors.append(mean - low)
                high_errors.append(high - mean)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=sorted_means[score_column],
                y=sorted_features,
                orientation="h",
                hoverinfo="x",
                marker=dict(color=colors),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=high_errors,
                    arrayminus=low_errors,
                ),
            )
        )
        fig.update_layout(
            margin=dict(l=200, b=0, t=40),
            xaxis=dict(
                title=title,
                range=[0, 1],
                showline=True,
                zeroline=False,
                side="top",
                fixedrange=True,
            ),
            yaxis=dict(showline=True, zeroline=False, fixedrange=True),
            plot_bgcolor="white",
        )
        return fig

    @classmethod
    def make_bias_ranking_fig(cls, ranking, **kwargs):
        return cls._make_ranking_fig(ranking, "importance", "Importance", **kwargs)

    @classmethod
    def make_performance_ranking_fig(cls, ranking, metric, criterion, **kwargs):
        return cls._make_ranking_fig(
            ranking, criterion, f"{criterion} {metric_to_col(metric)}", **kwargs
        )

    def _plot(self, explanation, make_fig, inline, **fig_kwargs):
        features = explanation["feature"].unique()
        if len(features) > 1:
            return plot(
                make_fig(explanation, with_taus=True, **fig_kwargs), inline=inline
            )
        return plot(
            make_fig(explanation, with_taus=False, **fig_kwargs)[features[0]],
            inline=inline,
        )

    def plot_bias(self, X_test, y_pred, inline=False, **fig_kwargs):
        explanation = self.explain_bias(X_test=X_test, y_pred=y_pred)
        return self._plot(explanation, self.make_bias_fig, inline=inline, **fig_kwargs)

    def plot_bias_ranking(self, X_test, y_pred, inline=False, **fig_kwargs):
        ranking = self.rank_by_bias(X_test=X_test, y_pred=y_pred)
        return plot(self.make_bias_ranking_fig(ranking, **fig_kwargs), inline=inline)

    def plot_performance(
        self, X_test, y_test, y_pred, metric, inline=False, **fig_kwargs
    ):
        explanation = self.explain_performance(
            X_test=X_test, y_test=y_test, y_pred=y_pred, metric=metric
        )
        return self._plot(
            explanation,
            functools.partial(self.make_performance_fig, metric=metric),
            inline=inline,
            **fig_kwargs,
        )

    def plot_performance_ranking(
        self, X_test, y_test, y_pred, metric, criterion, inline=False, **fig_kwargs
    ):
        ranking = self.rank_by_performance(
            X_test=X_test, y_test=y_test, y_pred=y_pred, metric=metric
        )
        return plot(
            self.make_performance_ranking_fig(
                ranking, criterion=criterion, metric=metric, **fig_kwargs
            ),
            inline=inline,
        )
