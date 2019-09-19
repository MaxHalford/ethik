import os

import pandas as pd

__all__ = ["load_law_school"]


def load_law_school():
    """Law school admission data.

    The Law School Admission Council conducted a survey across 163 law schools in the United
    States. It contains information on 21,790 law students such as their entrance exam scores
    (LSAT), their grade-point average (GPA) collected prior to law school, and their first year
    average grade (FYA).

    Given this data, a school may wish to predict if an applicant will have a high FYA. The school
    would also like to make sure these predictions are not biased by an individual’s race and sex.
    However, the LSAT, GPA, and FYA scores, may be biased due to social factors.

    Example:

        >>> import ethik

        >>> X, y = ethik.datasets.load_law_school()

        >>> X.head()
               race     sex  LSAT  UGPA region_first  ZFYA  sander_index
        0     White  Female  39.0   3.1           GL -0.98      0.782738
        1     White  Female  36.0   3.0           GL  0.09      0.735714
        2     White    Male  30.0   3.1           MS -0.35      0.670238
        5  Hispanic    Male  39.0   2.2           NE  0.58      0.697024
        6     White  Female  37.0   3.4           GL -1.26      0.786310

        >>> y.head()
        0    True
        1    True
        2    True
        5    True
        6    True
        Name: first_pf, dtype: bool

    References:
        1. https://papers.nips.cc/paper/6995-counterfactual-fairness.pdf
        2. https://github.com/mkusner/counterfactual-fairness

    """
    X = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "law_data.csv"),
        dtype={"first_pf": bool, "race": "category", "region_first": "category"},
        index_col=0
    )
    X["sex"] = X["sex"].map({1: "Female", 2: "Male"}).astype("category")
    y = X.pop("first_pf")
    return X, y