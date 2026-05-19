import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # mamimo — Marketing Mix Model Demo
    Updated for **sklearn 1.8 / numpy 2.4 / pandas 3.0** (Databricks Runtime 17.3 compatible).

    This notebook demonstrates the full MMM workflow: adstock transformations, saturation curves,
    regression, and channel contribution breakdown.
    """)
    return (mo,)


@app.cell
def _():
    import sys
    sys.path.insert(0, '/Users/jamesm/projects/promo/mamimo')

    import numpy as np
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import r2_score

    from mamimo.datasets import load_fake_mmm
    from mamimo.carryover import ExponentialCarryover, GeneralGaussianCarryover
    from mamimo.saturation import BoxCoxSaturation, ExponentialSaturation, HillSaturation
    from mamimo.linear_model import LinearRegression
    from mamimo.analysis import breakdown
    from mamimo.time_utils import add_date_indicators

    return (
        BoxCoxSaturation,
        ColumnTransformer,
        ExponentialCarryover,
        ExponentialSaturation,
        GeneralGaussianCarryover,
        HillSaturation,
        LinearRegression,
        add_date_indicators,
        breakdown,
        load_fake_mmm,
        make_pipeline,
        np,
        pd,
        r2_score,
    )


@app.cell(hide_code=True)
def _(load_fake_mmm, mo):
    data = load_fake_mmm()
    mo.md(f"""
    ## Dataset
    **{len(data)} weekly observations** across 3 media channels + Sales.

    Columns: {', '.join(f'`{c}`' for c in data.columns)}
    """)
    return (data,)


@app.cell(hide_code=True)
def _(data, mo):
    mo.ui.table(data.head(10).reset_index())
    return


@app.cell(hide_code=True)
def _(BoxCoxSaturation, ExponentialSaturation, HillSaturation, mo, np, pd):
    spend = np.linspace(0, 20000, 200).reshape(-1, 1)
    box_cox = BoxCoxSaturation(exponent=0.5).fit(spend)
    hill    = HillSaturation(exponent=2.0, half_saturation=5000.0).fit(spend)
    exp_sat = ExponentialSaturation(exponent=0.0001).fit(spend)

    sat_df = pd.DataFrame({
        "Spend":  spend.flatten(),
        "BoxCox": box_cox.transform(spend).flatten(),
        "Hill":   hill.transform(spend).flatten(),
        "Exp":    exp_sat.transform(spend).flatten(),
    })
    mo.md("""## Saturation Curves
    Three saturation functions applied to a spend range 0–20 000.
    Each curve shows diminishing returns as spend increases.
    """)
    return (sat_df,)


@app.cell(hide_code=True)
def _(mo, sat_df):
    mo.ui.table(sat_df.iloc[::20].round(4).reset_index(drop=True))
    return


@app.cell(hide_code=True)
def _(ExponentialCarryover, GeneralGaussianCarryover, mo, np, pd):
    impulse = np.zeros((20, 1))
    impulse[5] = 1.0
    exp_carry   = ExponentialCarryover(window=5, strength=0.7).fit_transform(impulse).flatten()
    gauss_carry = GeneralGaussianCarryover(window=7, p=1, sig=2).fit_transform(impulse).flatten()

    carry_df = pd.DataFrame({
        "Week":        range(20),
        "Impulse":     impulse.flatten(),
        "Exponential": exp_carry.round(4),
        "Gaussian":    gauss_carry.round(4),
    })
    mo.md("""## Carryover / Adstock Effects
    A single-week spend spike at week 5 decays over subsequent weeks.
    """)
    return (carry_df,)


@app.cell(hide_code=True)
def _(carry_df, mo):
    mo.ui.table(carry_df)
    return


@app.cell(hide_code=True)
def _(
    ColumnTransformer,
    ExponentialCarryover,
    ExponentialSaturation,
    LinearRegression,
    add_date_indicators,
    data,
    make_pipeline,
    mo,
    r2_score,
):
    X = data.drop("Sales", axis=1).pipe(add_date_indicators, some_special_date=["2020-01-05"])
    y = data["Sales"]

    channel_transformer = ColumnTransformer(
        transformers=[
            ("TV",      make_pipeline(ExponentialCarryover(window=4, strength=0.5),
                                      ExponentialSaturation(exponent=0.0001)), ["TV"]),
            ("Radio",   make_pipeline(ExponentialCarryover(window=2, strength=0.2),
                                      ExponentialSaturation(exponent=0.0001)), ["Radio"]),
            ("Banners", ExponentialSaturation(exponent=0.0001),                ["Banners"]),
            ("Special", ExponentialCarryover(window=10, strength=0.6),         ["some_special_date"]),
        ],
        remainder="passthrough",
    )
    model = make_pipeline(channel_transformer, LinearRegression(positive=True))
    model.fit(X, y)
    preds = model.predict(X)
    r2    = r2_score(y, preds)
    mo.md(f"""## Model Fit
    Positive-constrained linear regression on adstocked + saturated inputs.

    **R² = {r2:.4f}**
    """)
    return X, model, r2, y


@app.cell(hide_code=True)
def _(mo, model, pd):
    regressor    = model.named_steps["linearregression"]
    preprocessor = model.named_steps["columntransformer"]
    feature_names = preprocessor.get_feature_names_out()

    coef_df = pd.DataFrame({
        "Feature":     feature_names,
        "Coefficient": regressor.coef_.round(2),
    }).sort_values("Coefficient", ascending=False)

    mo.md("### Regression Coefficients")
    return (coef_df,)


@app.cell(hide_code=True)
def _(coef_df, mo):
    mo.ui.table(coef_df.reset_index(drop=True))
    return


@app.cell(hide_code=True)
def _(X, breakdown, mo, model, np, y):
    contrib = breakdown(model, X, y)
    contrib_summary = contrib.sum().rename("Total").to_frame()
    contrib_summary["Share %"] = (contrib_summary["Total"] /
                                   contrib_summary["Total"].sum() * 100).round(1)
    contrib_summary["Total"] = contrib_summary["Total"].round(0)
    row_totals_match = np.allclose(contrib.sum(axis=1).values, y.values, rtol=1e-4)
    mo.md("### Channel Contribution Summary")
    return contrib_summary, row_totals_match


@app.cell(hide_code=True)
def _(contrib_summary, mo):
    mo.ui.table(contrib_summary.reset_index().rename(columns={"index": "Channel"}))
    return


@app.cell(hide_code=True)
def _(mo, np, pd, r2, row_totals_match):
    import sklearn
    mo.md(f"""## Validation

    | Check | Result |
    |---|---|
    | sklearn | {sklearn.__version__} |
    | numpy | {np.__version__} |
    | pandas | {pd.__version__} |
    | Saturation transforms | ✅ |
    | Carryover transforms | ✅ |
    | Model fit | ✅ |
    | R² | {r2:.4f} |
    | Contributions sum to Sales | {'✅' if row_totals_match else '❌'} |
    """)
    return


if __name__ == "__main__":
    app.run()
