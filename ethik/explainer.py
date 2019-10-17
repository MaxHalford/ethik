import collections
import functools
import itertools
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy import special
from tqdm import tqdm

from .utils import decimal_range, join_with_overlap, to_pandas
from .warnings import ConstantWarning, ConvergenceWarning

__all__ = ["Explainer"]


CAT_COL_SEP = " = "


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
                    f"gradient descent failed to converge after {max_iterations} iterations "
                    + f"(name={x.name}, mean={mean}, target_mean={target_mean}, "
                    + f"current_mean={current_mean}, grad={grad}, hess={hess}, step={step}, ksi={ksi})"
                ),
                category=ConvergenceWarning,
            )

        ksis[(x.name, target_mean)] = ksi

    return ksis


def yield_masks(n_masks, n, p):
    """Generates a list of `n_masks` to keep a proportion `p` of `n` items.

    Args:
        n_masks (int): The number of masks to yield. It corresponds to the number
            of samples we use to compute the confidence interval.
        n (int): The number of items being filtered. It corresponds to the size
            of the dataset.
        p (float): The proportion of items to keep.

    Returns:
        generator: A generator of `n_masks` lists of `n` booleans being generated
            with a binomial distribution. As it is a probabilistic approach,
            we may get more or fewer than `p*n` items kept, but it is not a problem
            with large datasets.
    """

    if p < 0 or p > 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")

    if p < 1:
        for _ in range(n_masks):
            yield np.random.binomial(1, p, size=n).astype(bool)
    else:
        for _ in range(n_masks):
            yield np.full(shape=n, fill_value=True)


class Explainer:
    """Explains the influence and reliability of model predictions.

    Parameters:
        alpha (float): A `float` between `0` and `0.5` which indicates by how close the `Explainer`
            should look at extreme values of a distribution. The closer to zero, the more so
            extreme values will be accounted for. The default is `0.05` which means that all values
            beyond the 5th and 95th quantiles are ignored.
        n_taus (int): The number of τ values to consider. The results will be more fine-grained the
            higher this value is. However the computation time increases linearly with `n_taus`.
            The default is `41` and corresponds to each τ being separated by it's neighbors by
            `0.05`.
        n_samples (int): The number of samples to use for the confidence interval.
            If `1`, the default, no confidence interval is computed.
        sample_frac (float): The proportion of lines in the dataset sampled to
            generate the samples for the confidence interval. If `n_samples` is
            `1`, no confidence interval is computed and the whole dataset is used.
            Default is `0.8`.
        conf_level (float): A `float` between `0` and `0.5` which indicates the
            quantile used for the confidence interval. Default is `0.05`, which
            means that the confidence interval contains the data between the 5th
            and 95th quantiles.
        max_iterations (int): The maximum number of iterations used when applying the Newton step
            of the optimization procedure. Default is `5`.
        tol (float): The bottom threshold for the gradient of the optimization
            procedure. When reached, the procedure stops. Otherwise, a warning
            is raised about the fact that the optimization did not converge.
            Default is `1e-3`.
        n_jobs (int): The number of jobs to use for parallel computations. See
            `joblib.Parallel()`. Default is `-1`.
        memoize (bool): Indicates whether or not memoization should be used or not. If `True`, then
            intermediate results will be stored in order to avoid recomputing results that can be
            reused by successively called methods. For example, if you call `plot_influence` followed by
            `plot_influence_ranking` and `memoize` is `True`, then the intermediate results required by
            `plot_influence` will be reused for `plot_influence_ranking`. Memoization is turned off by
            default because it can lead to unexpected behavior depending on your usage.
        verbose (bool): Whether or not to show progress bars during
            computations. Default is `True`.
    """

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
        memoize=False,
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
        self.memoize = memoize
        self.verbose = verbose
        self.metric_names = set()
        self._reset_info()

    def _reset_info(self):
        """Resets the info dataframe (for when memoization is turned off)."""
        self.info = pd.DataFrame(
            columns=[
                "feature",
                "tau",
                "value",
                "ksi",
                "label",
                "influence",
                "influence_low",
                "influence_high",
            ]
        )

    def get_metric_name(self, metric):
        """Get the name of the column in explainer's info dataframe to store the
        performance with respect of the given metric.

        Args:
            metric (callable): The metric to compute the model's performance.

        Returns:
            str: The name of the column.
        """
        name = metric.__name__
        if name in self.info.columns and name not in self.metric_names:
            raise ValueError(f"Cannot use {name} as a metric name")
        return name

    @property
    def taus(self):
        tau_precision = 2 / (self.n_taus - 1)
        return list(decimal_range(-1, 1, tau_precision))

    @property
    def features(self):
        return self.info["feature"].unique().tolist()

    def _determine_pairs_to_do(self, features, labels):
        to_do_pairs = set(itertools.product(features, labels)) - set(
            self.info.groupby(["feature", "label"]).groups.keys()
        )
        to_do_map = collections.defaultdict(list)
        for feat, label in to_do_pairs:
            to_do_map[feat].append(label)
        return {feat: list(sorted(labels)) for feat, labels in to_do_map.items()}

    def _find_ksis(self, X_test, y_pred):
        """Finds ksi values for each (feature, tau, label) triplet.

        1. A list of $\\tau$ values is generated using `n_taus`. The $\\tau$ values range from -1 to 1.
        2. A grid of $\\eps$ values is generated for each $\\tau$ and for each variable. Each $\\eps$ represents a shift from a variable's mean towards a particular quantile.
        3. A grid of $\\ksi$ values is generated for each $\\eps$. Each $\\ksi$ corresponds to the optimal parameter that has to be used to weight the observations in order for the average to reach the associated $\\eps$ shift.

        Args:
            X_test (pandas.DataFrame or numpy.ndarray)
            y_pred (pandas.DataFrame or numpy.ndarray)
        """

        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        # One-hot encode the categorical features
        X_test = pd.get_dummies(data=X_test, prefix_sep=CAT_COL_SEP)

        # Check which (feature, label) pairs have to be done
        to_do_map = self._determine_pairs_to_do(
            features=X_test.columns, labels=y_pred.columns
        )
        # We need a list to keep the order of X_test
        to_do_features = list(feat for feat in X_test.columns if feat in to_do_map)
        X_test = X_test[to_do_features]

        if X_test.empty:
            return self

        # Make the epsilons for each (feature, label, tau) triplet
        quantiles = X_test.quantile(q=[self.alpha, 1.0 - self.alpha])

        # Issue a warning if a feature doesn't have distinct quantiles
        for feature, n_unique in quantiles.nunique().to_dict().items():
            if n_unique == 1:
                warnings.warn(
                    message=f"all the values of feature {feature} are identical",
                    category=ConstantWarning,
                )

        q_mins = quantiles.loc[self.alpha].to_dict()
        q_maxs = quantiles.loc[1.0 - self.alpha].to_dict()
        means = X_test.mean().to_dict()
        additional_info = pd.concat(
            [
                pd.DataFrame(
                    {
                        "tau": self.taus,
                        "value": [
                            means[feature]
                            + tau
                            * (
                                max(means[feature] - q_mins[feature], 0)
                                if tau < 0
                                else max(q_maxs[feature] - means[feature], 0)
                            )
                            for tau in self.taus
                        ],
                        "feature": [feature] * len(self.taus),
                        "label": [label] * len(self.taus),
                    }
                )
                # We need to iterate over `to_do_features` to keep the order of X_test
                for feature in to_do_features
                for label in to_do_map[feature]
            ],
            ignore_index=True,
        )
        self.info = self.info.append(additional_info, ignore_index=True, sort=False)

        # Find a ksi for each (feature, espilon) pair
        ksis = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(compute_ksis)(
                x=X_test[feature],
                target_means=part["value"].unique(),
                max_iterations=self.max_iterations,
                tol=self.tol,
            )
            for feature, part in self.info.groupby("feature")
            if feature in to_do_features
        )
        ksis = dict(collections.ChainMap(*ksis))
        self.info["ksi"] = self.info.apply(
            lambda r: ksis.get((r["feature"], r["value"]), r["ksi"]), axis="columns"
        )
        self.info["ksi"] = self.info["ksi"].fillna(0.0)

        return self

    def _explain(self, X_test, y_pred, dest_col, key_cols, compute):

        # Reset info if memoization is turned off
        if not self.memoize:
            self._reset_info()
        if dest_col not in self.info.columns:
            self.info[dest_col] = None
            self.info[f"{dest_col}_low"] = None
            self.info[f"{dest_col}_high"] = None

        # Coerce X_test and y_pred to DataFrames
        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        # Check X_test and y_pred okay
        if len(X_test) != len(y_pred):
            raise ValueError("X_test and y_pred are not of the same length")

        # Find the ksi values for each (feature, tau, label) triplet
        self._find_ksis(X_test, y_pred)

        # One-hot encode the categorical variables
        X_test = pd.get_dummies(data=X_test, prefix_sep=CAT_COL_SEP)

        # Determine which features are missing explanations; that is they have null influences for at
        # least one ksi value
        relevant = self.info[
            self.info["feature"].isin(X_test.columns)
            & self.info["label"].isin(y_pred.columns)
            & self.info[dest_col].isnull()
        ]

        if not relevant.empty:
            # `compute()` will return something like:
            # [
            #   [ # First batch
            #     (*key_cols1, sample_index1, computed_value1),
            #     (*key_cols2, sample_index2, computed_value2),
            #     ...
            #   ],
            #   ...
            # ]
            data = compute(X_test=X_test, y_pred=y_pred, relevant=relevant)
            # We group by the key to gather the samples and compute the confidence
            #  interval
            data = data.groupby(key_cols)[dest_col].agg(
                [
                    # Mean influence
                    (dest_col, "mean"),
                    # Lower bound on the mean influence
                    (
                        f"{dest_col}_low",
                        functools.partial(np.quantile, q=self.conf_level),
                    ),
                    # Upper bound on the mean influence
                    (
                        f"{dest_col}_high",
                        functools.partial(np.quantile, q=1 - self.conf_level),
                    ),
                ]
            )

            # Merge the new information with the current information
            self.info = join_with_overlap(left=self.info, right=data, on=key_cols)

        return self.info[
            self.info["feature"].isin(X_test.columns)
            & self.info["label"].isin(y_pred.columns)
        ]

    def _explain_individual(self, X_test, y_pred, individual, dest_col, compute):
        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))
        individual = to_pandas(individual)  #  individual is a pd.Series now

        # One-hot encode the categorical variables
        X_test = pd.get_dummies(data=X_test, prefix_sep=CAT_COL_SEP)
        individual = pd.get_dummies(
            data=pd.DataFrame(  #  A DataFrame is needed for get_dummies
                [individual.values], columns=individual.index
            ),
            prefix_sep=CAT_COL_SEP,
        ).iloc[0]

        explanation = collections.defaultdict(list)

        for feature in X_test:
            if feature not in individual:
                continue

            ksi = compute_ksis(
                x=X_test[feature],
                target_means=[individual[feature]],
                max_iterations=self.max_iterations,
                tol=self.tol,
            )
            ksi = list(ksi.values())[0]

            mean_explanation = compute(X_test, y_pred, feature, 0)
            individual_explanation = compute(X_test, y_pred, feature, ksi)

            for mean_entry, individual_entry in zip(
                mean_explanation, individual_explanation
            ):
                individual_entry["delta_to_mean"] = (
                    individual_entry[dest_col] - mean_entry[dest_col]
                )
                for k, v in individual_entry.items():
                    explanation[k].append(v)

        return pd.DataFrame(explanation)

    def explain_influence(self, X_test, y_pred, individual=None):
        """Compute the influence of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or pd.Series): The dataset as a pandas dataframe
                with one column per feature or a pandas series for a single feature.
            y_pred (pd.DataFrame or pd.Series): The model predictions
                for the samples in `X_test`. For binary classification and regression,
                `pd.Series` is expected. For multi-label classification, a
                pandas dataframe with one column per label is
                expected. The values can either be probabilities or `0/1`
                (for a one-hot-encoded output).

        Returns:
            pd.DataFrame:
                A dataframe with columns `(feature, tau, value, ksi, label,
                influence, influence_low, influence_high)`. If `explainer.n_samples` is `1`,
                no confidence interval is computed and `influence = influence_low = influence_high`.
                The value of `label` is not important for regression.

        Examples:
            See more examples in `notebooks`.

            Binary classification:

            >>> X_test = pd.DataFrame([
            ...     [1, 2],
            ...     [1.1, 2.2],
            ...     [1.3, 2.3],
            ... ], columns=["x0", "x1"])
            >>> y_pred = model.predict(X_test)
            >>> y_pred
            [0, 1, 1]  # Can also be probabilities: [0.3, 0.65, 0.8]
            >>> # For readibility reasons, we give a name to the predictions
            >>> y_pred = pd.Series(y_pred, name="is_reliable")
            >>> explainer.explain_influence(X_test, y_pred)

            Regression is similar to binary classification:

            >>> X_test = pd.DataFrame([
            ...     [1, 2],
            ...     [1.1, 2.2],
            ...     [1.3, 2.3],
            ... ], columns=["x0", "x1"])
            >>> y_pred = model.predict(X_test)
            >>> y_pred
            [22, 24, 19]
            >>> # For readibility reasons, we give a name to the predictions
            >>> y_pred = pd.Series(y_pred, name="price")
            >>> explainer.explain_influence(X_test, y_pred)

            For multi-label classification, we need a dataframe to store predictions:

            >>> X_test = pd.DataFrame([
            ...     [1, 2],
            ...     [1.1, 2.2],
            ...     [1.3, 2.3],
            ... ], columns=["x0", "x1"])
            >>> y_pred = model.predict(X_test)
            >>> y_pred.columns
            ["class0", "class1", "class2"]
            >>> y_pred.iloc[0]
            [0, 1, 0] # One-hot encoded, or probabilities: [0.15, 0.6, 0.25]
            >>> explainer.explain_influence(X_test, y_pred)
        """

        def compute_individual(X_test, y_pred, feature, label, ksi, mask=None):
            if mask is None:
                mask = np.full(shape=len(X_test), fill_value=True)

            return np.average(
                y_pred[label][mask],
                weights=special.softmax(ksi * X_test[feature][mask]),
            )

        def compute(X_test, y_pred, relevant):
            keys = relevant.groupby(["feature", "label", "ksi"]).groups.keys()
            return pd.DataFrame(
                [
                    (
                        feature,
                        label,
                        ksi,
                        sample_index,
                        compute_individual(
                            X_test, y_pred, feature, label, ksi, mask=mask
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

        if individual is not None:
            return self._explain_individual(
                X_test=X_test,
                y_pred=y_pred,
                individual=individual,
                dest_col="influence",
                compute=lambda X_test, y_pred, feature, ksi: [
                    dict(
                        label=label,
                        feature=feature,
                        influence=compute_individual(
                            X_test, y_pred, feature, label, ksi
                        ),
                    )
                    for label in y_pred.columns
                ],
            )

        return self._explain(
            X_test=X_test,
            y_pred=y_pred,
            dest_col="influence",
            key_cols=["feature", "label", "ksi"],
            compute=compute,
        )

    def explain_performance(self, X_test, y_test, y_pred, metric, individual=None):
        """Compute the change in model's performance for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or pd.Series): The dataset as a pandas dataframe
                with one column per feature or a pandas series for a single feature.
            y_test (pd.DataFrame or pd.Series): The true values
                for the samples in `X_test`. For binary classification and regression,
                a `pd.Series` is expected. For multi-label classification,
                a pandas dataframe with one column per label is
                expected. The values can either be probabilities or `0/1`
                (for a one-hot-encoded output).
            y_pred (pd.DataFrame or pd.Series): The model predictions
                for the samples in `X_test`. The format is the same as `y_test`.
            metric (callable): A scikit-learn-like metric
                `f(y_true, y_pred, sample_weight=None)`. The metric must be able
                to handle the `y` data. For instance, for `sklearn.metrics.accuracy_score()`,
                "the set of labels predicted for a sample must exactly match the
                corresponding set of labels in `y_true`".

        Returns:
            pd.DataFrame:
                A dataframe with columns `(feature, tau, value, ksi, label,
                influence, influence_low, influence_high, <metric_name>, <metric_name_low>, <metric_name_high>)`.
                If `explainer.n_samples` is `1`, no confidence interval is computed
                and `<metric_name> = <metric_name_low> = <metric_name_high>`.
                The value of `label` is not important for regression.

        Examples:
            See examples in `notebooks`.
        """
        metric_name = self.get_metric_name(metric)
        if metric_name not in self.info.columns:
            self.info[metric_name] = None
            self.info[f"{metric_name}_low"] = None
            self.info[f"{metric_name}_high"] = None
        self.metric_names.add(metric_name)

        y_test = np.asarray(y_test)

        def compute_individual(X_test, y_pred, feature, ksi, mask=None):
            if mask is None:
                mask = np.full(shape=len(X_test), fill_value=True)

            return metric(
                y_test[mask],
                y_pred[mask],
                sample_weight=special.softmax(ksi * X_test[feature][mask]),
            )

        def compute(X_test, y_pred, relevant):
            keys = relevant.groupby(["feature", "ksi"]).groups.keys()
            return pd.DataFrame(
                [
                    (
                        feature,
                        ksi,
                        sample_index,
                        compute_individual(X_test, y_pred, feature, ksi, mask=mask),
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

        if individual is not None:
            return self._explain_individual(
                X_test=X_test,
                y_pred=y_pred,
                individual=individual,
                dest_col=metric_name,
                compute=lambda X_test, y_pred, feature, ksi: [
                    {
                        "feature": feature,
                        metric_name: compute_individual(X_test, y_pred, feature, ksi),
                    }
                ],
            )

        return self._explain(
            X_test=X_test,
            y_pred=y_pred,
            dest_col=metric_name,
            key_cols=["feature", "ksi"],
            compute=compute,
        )

    def rank_by_influence(self, X_test, y_pred):
        """Returns a pandas DataFrame containing the importance of each feature
        per label.

        Args:
            X_test (pd.DataFrame or pd.Series): The dataset as a pandas dataframe
                with one column per feature or a pandas series for a single feature.
            y_pred (pd.DataFrame or pd.Series): The model predictions
                for the samples in `X_test`. For binary classification and regression,
                a `pd.Series` is expected. For multi-label classification,
                a pandas dataframe with one column per label is
                expected. The values can either be probabilities or `0/1`
                (for a one-hot-encoded output).

        Returns:
            pd.DataFrame:
                A dataframe with columns `(label, feature, importance)`. The row
                `(setosa, petal length (cm), 0.282507)` means that the feature
                `petal length` of the Iris dataset has an importance of about
                30% in the prediction of the class `setosa`.

                The importance is a real number between 0 and 1. Intuitively,
                if the model influence for the feature `X` is a flat curve (the average
                model prediction is not impacted by the mean of `X`) then we
                can conclude that `X` has no importance for predictions. This
                flat curve is the baseline and satisfies \\(y = influence_{\\tau(0)}\\).
                To compute the importance of a feature, we look at the average
                distance of the influence curve to this baseline:

                $$
                I(X) = \\frac{1}{n_\\tau} \\sum_{i=1}^{n_\\tau} \\mid influence_{\\tau(i)}(X) - influence_{\\tau(0)}(X) \\mid
                $$

                The influence curve is first normalized so that the importance is
                between 0 and 1 (which may not be the case originally for regression
                problems). To normalize, we get the minimum and maximum influences
                *across all features and all classes* and then compute
                `normalized = (influence - min) / (max - min)`.

                For regression problems, there's one label only and its name
                doesn't matter (it's just to have a consistent output).
        """

        def get_importance(group, min_influence, max_influence):
            """Computes the average absolute difference in influence changes per tau increase."""
            #  Normalize influence to get an importance between 0 and 1
            # influence can be outside [0, 1] for regression
            influence = group["influence"]
            group["influence"] = (influence - min_influence) / (
                max_influence - min_influence
            )
            baseline = group.query("tau == 0").iloc[0]["influence"]
            return (group["influence"] - baseline).abs().mean()

        explanation = self.explain_influence(X_test=X_test, y_pred=y_pred)
        min_influence = explanation["influence"].min()
        max_influence = explanation["influence"].max()

        return (
            explanation.groupby(["label", "feature"])
            .apply(
                functools.partial(
                    get_importance,
                    min_influence=min_influence,
                    max_influence=max_influence,
                )
            )
            .to_frame("importance")
            .reset_index()
        )

    def rank_by_performance(self, X_test, y_test, y_pred, metric):
        """Returns a pandas DataFrame containing
        per label.

        Args:
            X_test (pd.DataFrame or pd.Series): The dataset as a pandas dataframe
                with one column per feature or a pandas series for a single feature.
            y_test (pd.DataFrame or pd.Series): The true output
                for the samples in `X_test`. For binary classification and regression,
                a `pd.Series` is expected. For multi-label classification,
                a pandas dataframe with one column per label is
                expected. The values can either be probabilities or `0/1`
                (for a one-hot-encoded output).
            y_pred (pd.DataFrame or pd.Series): The model predictions
                for the samples in `X_test`. The format is the same as `y_test`.
            metric (callable): A scikit-learn-like metric
                `f(y_true, y_pred, sample_weight=None)`. The metric must be able
                to handle the `y` data. For instance, for `sklearn.metrics.accuracy_score()`,
                "the set of labels predicted for a sample must exactly match the
                corresponding set of labels in `y_true`".

        Returns:
            pd.DataFrame:
                A dataframe with columns `(feature, min, max)`. The row
                `(age, 0.862010, 0.996360)` means that the score measured by the
                given metric (e.g. `sklearn.metrics.accuracy_score`) stays bewteen
                86.2% and 99.6% on average when we make the mean age change. With
                such information, we can find the features for which the model
                performs the worst or the best.

                For regression problems, there's one label only and its name
                doesn't matter (it's just to have a consistent output).
        """
        metric_name = self.get_metric_name(metric)

        def get_aggregates(df):
            return pd.Series(
                [df[metric_name].min(), df[metric_name].max()], index=["min", "max"]
            )

        return (
            self.explain_performance(X_test, y_test, y_pred, metric)
            .groupby("feature")
            .apply(get_aggregates)
            .reset_index()
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
                taus = explanation.loc[explanation["feature"] == feat, "tau"]
                values = explanation.loc[explanation["feature"] == feat, "value"]
                y = explanation.loc[explanation["feature"] == feat, col]
                fig.add_trace(
                    go.Scatter(
                        x=taus,
                        y=y,
                        mode="lines+markers",
                        hoverinfo="y",
                        name=feat,
                        customdata=list(zip(taus, values)),
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

        # There is only one feature, we plot it with its nominal values.
        feat = features[0]
        fig = go.Figure()
        x = explanation.loc[explanation["feature"] == feat, "value"]
        y = explanation.loc[explanation["feature"] == feat, col]
        mean_row = explanation[
            (explanation["feature"] == feat) & (explanation["tau"] == 0)
        ].iloc[0]

        if self.n_samples > 1:
            low = explanation.loc[explanation["feature"] == feat, f"{col}_low"]
            high = explanation.loc[explanation["feature"] == feat, f"{col}_high"]
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
                x=[mean_row["value"]],
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

    def _plot_individual_explanation(
        self, explanation, title, colors=None, size=None, yrange=None
    ):
        explanation = explanation.sort_values(by=["delta_to_mean"], ascending=True)
        features = explanation["feature"]

        if colors is None:
            colors = {}
        elif type(colors) is str:
            colors = {feat: colors for feat in features}

        width = 500
        height = 100 + 60 * len(features)
        if size is not None:
            width, height = size

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=explanation["delta_to_mean"],
                y=features,
                orientation="h",
                hoverinfo="x",
            )
        )
        fig.update_layout(
            xaxis=dict(
                title=f"Difference in {title} with original dataset",
                range=yrange,
                showline=True,
                linewidth=1,
                linecolor="black",
                zeroline=False,
                gridcolor="#eee",
                side="top",
                fixedrange=True,
            ),
            yaxis=dict(
                showline=False,
                zeroline=False,
                fixedrange=True,
                linecolor="black",
                automargin=True,
            ),
            shapes=[
                go.layout.Shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="black", width=1),
                )
            ],
            plot_bgcolor="white",
            width=width,
            height=height,
        )
        return fig

    def _plot_ranking(
        self, ranking, score_column, title, n_features=None, colors=None, size=None
    ):
        if n_features is None:
            n_features = len(ranking)
        ascending = n_features >= 0
        ranking = ranking.sort_values(by=[score_column], ascending=ascending)
        n_features = abs(n_features)

        width = 500
        height = 50 * n_features
        if size is not None:
            width, height = size

        return go.Figure(
            data=[
                go.Bar(
                    x=ranking[score_column][-n_features:],
                    y=ranking["feature"][-n_features:],
                    orientation="h",
                    hoverinfo="x",
                    marker=dict(color=colors),
                )
            ],
            layout=go.Layout(
                margin=dict(b=0, t=40),
                width=width,
                height=height,
                xaxis=dict(
                    title=title,
                    range=[0, 1],
                    showline=True,
                    linecolor="black",
                    linewidth=1,
                    zeroline=False,
                    gridcolor="#eee",
                    side="top",
                    fixedrange=True,
                ),
                yaxis=dict(
                    showline=True,
                    linecolor="black",
                    linewidth=1,
                    zeroline=False,
                    fixedrange=True,
                    automargin=True,
                ),
                plot_bgcolor="white",
            ),
        )

    def plot_influence(
        self, X_test, y_pred, individual=None, colors=None, yrange=None, size=None
    ):
        """Plot the influence of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or np.array): See `Explainer.explain_influence()`.
            y_pred (pd.DataFrame or pd.Series): See `Explainer.explain_influence()`.
            colors (dict, optional): A dictionary that maps features to colors.
                Default is `None` and the colors are choosen automatically.
            yrange (list, optional): A two-item list `[low, high]`. Default is
                `None` and the range is based on the data.

        Returns:
            plotly.graph_objs.Figure:
                A Plotly figure. It shows automatically in notebook cells but you
                can also call the `.show()` method to plot multiple charts in the
                same cell.

        Examples:
            >>> explainer.plot_influence(X_test, y_pred)
            >>> explainer.plot_influence(X_test, y_pred, colors=dict(
            ...     x0="blue",
            ...     x1="red",
            ... ))
            >>> explainer.plot_influence(X_test, y_pred, yrange=[0.5, 1])
        """
        explanation = self.explain_influence(X_test, y_pred, individual=individual)
        labels = explanation["label"].unique()
        if len(labels) > 1:
            raise ValueError("Cannot plot multiple labels")

        if individual is not None:
            return self._plot_individual_explanation(
                explanation, title="influence", colors=colors, size=size, yrange=yrange
            )

        y_label = f"Average '{labels[0]}'"
        return self._plot_explanation(
            explanation, "influence", y_label, colors=colors, yrange=yrange, size=size
        )

    def plot_influence_ranking(self, X_test, y_pred, n_features=None, colors=None):
        """Plot the ranking of the features based on their influence.

        Args:
            X_test (pd.DataFrame or np.array): See `Explainer.explain_influence()`.
            y_pred (pd.DataFrame or pd.Series): See `Explainer.explain_influence()`.
            n_features (int, optional): The number of features to plot. With the
                default (`None`), all of them are shown. For a positive value,
                we keep the `n_features` first features (the most impactful). For
                a negative value, we keep the `n_features` last features.
            colors (dict, optional): See `Explainer.plot_influence()`.

        Returns:
            plotly.graph_objs.Figure:
                A Plotly figure. It shows automatically in notebook cells but you
                can also call the `.show()` method to plot multiple charts in the
                same cell.
        """
        ranking = self.rank_by_influence(X_test=X_test, y_pred=y_pred)
        return self._plot_ranking(
            ranking=ranking,
            score_column="importance",
            title="Importance",
            n_features=n_features,
            colors=colors,
        )

    def plot_performance(
        self,
        X_test,
        y_test,
        y_pred,
        metric,
        individual=None,
        colors=None,
        yrange=None,
        size=None,
    ):
        """Plot the performance of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or np.array): See `Explainer.explain_performance()`.
            y_test (pd.DataFrame or pd.Series): See `Explainer.explain_performance()`.
            y_pred (pd.DataFrame or pd.Series): See `Explainer.explain_performance()`.
            metric (callable): See `Explainer.explain_performance()`.
            colors (dict, optional): See `Explainer.plot_influence()`.
            yrange (list, optional): See `Explainer.plot_influence()`.

        Returns:
            plotly.graph_objs.Figure:
                A Plotly figure. It shows automatically in notebook cells but you
                can also call the `.show()` method to plot multiple charts in the
                same cell.
        """
        metric_name = self.get_metric_name(metric)
        explanation = self.explain_performance(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            metric=metric,
            individual=individual,
        )
        if yrange is None:
            if explanation[metric_name].between(0, 1).all():
                yrange = [0, 1] if individual is None else [-1, 1]

        if individual is not None:
            return self._plot_individual_explanation(
                explanation, title=metric_name, colors=colors, size=size, yrange=yrange
            )

        return self._plot_explanation(
            explanation,
            metric_name,
            y_label=f"Average {metric_name}",
            colors=colors,
            yrange=yrange,
            size=size,
        )

    def plot_performance_ranking(
        self, X_test, y_test, y_pred, metric, criterion, n_features=None, colors=None
    ):
        """Plot the performance of the model for the features in `X_test`.

        Args:
            X_test (pd.DataFrame or np.array): See `Explainer.explain_performance()`.
            y_test (pd.DataFrame or pd.Series): See `Explainer.explain_performance()`.
            y_pred (pd.DataFrame or pd.Series): See `Explainer.explain_performance()`.
            metric (callable): See `Explainer.explain_performance()`.
            criterion (str): Either "min" or "max" to determine whether, for a
                given feature, we keep the worst or the best performance for all
                the values taken by the mean. See `Explainer.rank_by_performance()`.
            n_features (int, optional): The number of features to plot. With the
                default (`None`), all of them are shown. For a positive value,
                we keep the `n_features` first features (the most impactful). For
                a negative value, we keep the `n_features` last features.
            colors (dict, optional): See `Explainer.plot_influence_ranking()`.

        Returns:
            plotly.graph_objs.Figure:
                A Plotly figure. It shows automatically in notebook cells but you
                can also call the `.show()` method to plot multiple charts in the
                same cell.
        """
        metric_name = self.get_metric_name(metric)
        ranking = self.rank_by_performance(
            X_test=X_test, y_test=y_test, y_pred=y_pred, metric=metric
        )
        return self._plot_ranking(
            ranking=ranking,
            score_column=criterion,
            title=f"{criterion} {metric_name}",
            n_features=n_features,
            colors=colors,
        )
