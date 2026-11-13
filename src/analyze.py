import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from train import FEATURES, TARGET, build_forest, build_linear, load_splits


def main():
    train, val, _ = load_splits()  # test set stays untouched
    val = val.copy()

    for name, pipe in [("linear", build_linear()), ("forest", build_forest())]:
        pipe.fit(train[FEATURES], train[TARGET])
        val[f"pred_{name}"] = pipe.predict(val[FEATURES])
        val[f"err_{name}"] = val[f"pred_{name}"] - val[TARGET]
        val[f"abs_{name}"] = val[f"err_{name}"].abs()

    # 1. Actual vs predicted, all series summed per day
    daily = val.groupby("date")[[TARGET, "pred_linear", "pred_forest"]].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    daily[TARGET].plot(ax=ax, label="Actual", linewidth=1.4)
    daily["pred_linear"].plot(ax=ax, label="Linear regression", linewidth=1)
    daily["pred_forest"].plot(ax=ax, label="Random forest", linewidth=1)
    ax.set(title="Validation period: actual vs predicted (daily totals)", ylabel="Units")
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/figures/06_val_actual_vs_pred.png", dpi=150)
    plt.close(fig)

    # 2. Bias by month: negative = under-predicting, positive = over-predicting
    month = val["date"].dt.month
    
    print("Average error (prediction - actual) per row, by month")
    print(val.groupby(month)[["err_linear", "err_forest"]].mean().round(2))

    # 3. Accuracy by weekday
    print("\nMAE by day of week (0 = Monday)")
    print(val.groupby("dow")[["abs_linear", "abs_forest"]].mean().round(2))


if __name__ == "__main__":
    main()