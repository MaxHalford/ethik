import colorlover as cl
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from .explainer import Explainer
from .utils import to_pandas

__all__ = ["ClassificationExplainer"]


class ClassificationExplainer(Explainer):
    def _plot_individual_influence(
        self, X_test, y_pred, individual, colors=None, yrange=None, size=None
    ):
        if yrange is None:
            yrange = [-1, 1]

        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        features = X_test.columns
        labels = y_pred.columns
        plots = []

        if len(labels) == 1:
            return super().plot_influence(
                X_test,
                y_pred.iloc[:, 0],
                individual=individual,
                colors=colors,
                yrange=yrange,
                size=size,
            )

        if colors is None:
            #  Skip the lightest color as it is too light
            scale = cl.interp(cl.scales["10"]["qual"]["Paired"], len(features) + 1)[1:]
            colors = {feat: scale[i] for i, feat in enumerate(features)}

        for label in labels:
            plots.append(
                super().plot_influence(
                    X_test,
                    y_pred[label],
                    individual=individual,
                    colors=colors,
                    yrange=yrange,
                )
            )

        fig = make_subplots(rows=len(labels), cols=1, shared_xaxes=True)
        shapes = []
        for ilabel, (label, plot) in enumerate(zip(labels, plots)):
            for trace in plot["data"]:
                fig.update_layout({f"yaxis{ilabel+1}": dict(title=label)})
                shapes.append(
                    go.layout.Shape(
                        type="line",
                        x0=0,
                        y0=-0.5,
                        x1=0,
                        y1=len(features) - 0.5,
                        yref=f"y{ilabel+1}",
                        line=dict(color="black", width=1),
                    )
                )
                fig.add_trace(trace, row=ilabel + 1, col=1)

        width = height = None
        if size is not None:
            width, height = size

        """
        fig.update_xaxes(
            nticks=5,
            showline=True,
            showgrid=True,
            zeroline=False,
            linecolor="black",
            gridcolor="#eee",
        )
        """
        fig.update_xaxes(
            range=yrange,
            showline=True,
            linecolor="black",
            linewidth=1,
            zeroline=False,
            showgrid=True,
            gridcolor="#eee",
            side="top",
            fixedrange=True,
            showticklabels=True,
        )
        fig.update_layout(
            showlegend=False,
            xaxis1=dict(title=f"Difference in influence with original dataset"),
            shapes=shapes,
            plot_bgcolor="white",
            width=width,
            height=height,
        )
        return fig

    def plot_influence(
        self, X_test, y_pred, individual=None, colors=None, yrange=None, size=None
    ):
        """Plot the influence for the features in `X_test`.

        See `ethik.explainer.Explainer.plot_influence()`.
        """
        if individual is not None:
            return self._plot_individual_influence(
                X_test, y_pred, individual, colors=colors, yrange=yrange, size=size
            )

        if yrange is None:
            yrange = [0, 1]

        X_test = pd.DataFrame(to_pandas(X_test))
        y_pred = pd.DataFrame(to_pandas(y_pred))

        if len(y_pred.columns) == 1:
            return super().plot_influence(
                X_test, y_pred.iloc[:, 0], colors=colors, yrange=yrange, size=size
            )

        if colors is None:
            features = X_test.columns
            #  Skip the lightest color as it is too light
            scale = cl.interp(cl.scales["10"]["qual"]["Paired"], len(features) + 1)[1:]
            colors = {feat: scale[i] for i, feat in enumerate(features)}

        labels = y_pred.columns
        plots = []
        for label in labels:
            plots.append(
                super().plot_influence(
                    X_test, y_pred[label], colors=colors, yrange=yrange
                )
            )

        fig = make_subplots(rows=len(labels), cols=1, shared_xaxes=True)
        for ilabel, (label, plot) in enumerate(zip(labels, plots)):
            fig.update_layout({f"yaxis{ilabel+1}": dict(title=f"Average {label}")})
            for trace in plot["data"]:
                trace["showlegend"] = ilabel == 0 and trace["showlegend"]
                trace["legendgroup"] = trace["name"]
                fig.add_trace(trace, row=ilabel + 1, col=1)

        width = height = None
        if size is not None:
            width, height = size

        fig.update_xaxes(
            nticks=5,
            showline=True,
            showgrid=True,
            zeroline=False,
            linecolor="black",
            gridcolor="#eee",
        )
        fig.update_yaxes(
            range=yrange,
            showline=True,
            showgrid=True,
            linecolor="black",
            gridcolor="#eee",
        )
        fig.update_layout(
            {
                f"xaxis{len(labels)}": dict(
                    title="tau"
                    if len(X_test.columns) > 1
                    else f"Average {X_test.columns[0]}"
                ),
                "plot_bgcolor": "white",
                "width": width,
                "height": height,
            }
        )
        return fig
