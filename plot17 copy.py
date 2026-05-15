# Kompletny skrypt analizy hiperparametrów

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ==========================================================
# KONFIGURACJA
# ==========================================================

MLRUNS_PATH = "mlruns/0"
OUTPUT_FILE = "hyperparameter_analysis_sw.xlsx"

# ==========================================================
# WCZYTANIE WYNIKÓW Z MLRUNS
# ==========================================================

records = []

for run_id in os.listdir(MLRUNS_PATH):
    run_path = os.path.join(MLRUNS_PATH, run_id)

    if not os.path.isdir(run_path):
        continue

    try:
        params_path = os.path.join(run_path, "params")
        metrics_path = os.path.join(run_path, "metrics")

        required_params = [
            "window_size",
            "epochs",
            "learning_rate",
            "random_seed",
        ]

        params = {}

        for p in required_params:
            p_file = os.path.join(params_path, p)

            if not os.path.exists(p_file):
                raise ValueError("missing param")

            with open(p_file) as f:
                params[p] = f.read().strip()

        accuracy_file = os.path.join(metrics_path, "accuracy")

        if not os.path.exists(accuracy_file):
            continue

        with open(accuracy_file) as f:
            lines = f.readlines()
            last = lines[-1].strip().split()
            accuracy = float(last[1])

        record = {
            "window_size": int(params["window_size"]),
            "epochs": float(params["epochs"]),
            "learning_rate": float(params["learning_rate"]),
            "random_seed": int(params["random_seed"]),
            "accuracy": accuracy,
        }

        records.append(record)

    except Exception:
        continue

# ==========================================================
# DATAFRAME Z DANYMI
# ==========================================================

df = pd.DataFrame(records)
print("Runów poprawnych:", len(df))

if df.empty:
    raise ValueError("Nie znaleziono poprawnych runów.")

# ==========================================================
# UŚREDNIENIE PO RANDOM_SEED
# ==========================================================

# Dla każdej kombinacji (window_size, epochs, learning_rate)
# obliczamy średnią accuracy.

df_mean = (
    df.groupby(
        ["window_size", "epochs", "learning_rate"],
        as_index=False,
    )["accuracy"]
    .mean()
)

# ==========================================================
# 1. TWORZENIE TABEL PIVOT (jak w Twoim skrypcie)
# ==========================================================

pivot_tables = {}

for lr in sorted(df_mean["learning_rate"].unique()):
    pivot = (
        df_mean[df_mean["learning_rate"] == lr]
        .pivot(
            index="window_size",
            columns="epochs",
            values="accuracy",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    pivot_tables[lr] = pivot

# ==========================================================
# 2. ANALIZA WPŁYWU PARAMETRÓW
# ==========================================================

parameters = [
    "window_size",
    "epochs",
    "learning_rate",
]

analysis_tables = {}
summary_rows = []

for param in parameters:
    # ------------------------------------------------------
    # Średnia accuracy dla każdej wartości parametru
    # ------------------------------------------------------
    stats = (
        df_mean.groupby(param)["accuracy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(param)
    )

    analysis_tables[f"{param}_means"] = stats

    # ------------------------------------------------------
    # effect_range = max(mean) - min(mean)
    # ------------------------------------------------------
    effect_range = stats["mean"].max() - stats["mean"].min()

    # ------------------------------------------------------
    # Korelacja Spearmana
    # ------------------------------------------------------
    corr, p_value = spearmanr(df_mean[param], df_mean["accuracy"])

    # ------------------------------------------------------
    # ANOVA jednoczynnikowa
    # ------------------------------------------------------
    model = ols(f"accuracy ~ C({param})", data=df_mean).fit()
    anova = sm.stats.anova_lm(model, typ=2)

    analysis_tables[f"{param}_anova"] = anova.reset_index()

    # ------------------------------------------------------
    # Eta squared
    # ------------------------------------------------------
    ss_param = anova.loc[f"C({param})", "sum_sq"]
    ss_total = anova["sum_sq"].sum()
    eta_squared = ss_param / ss_total

    # ------------------------------------------------------
    # Najlepsza i najgorsza wartość parametru
    # ------------------------------------------------------
    best_idx = stats["mean"].idxmax()
    worst_idx = stats["mean"].idxmin()

    best_value = stats.loc[best_idx, param]
    best_mean_accuracy = stats.loc[best_idx, "mean"]

    worst_value = stats.loc[worst_idx, param]
    worst_mean_accuracy = stats.loc[worst_idx, "mean"]

    # ------------------------------------------------------
    # Wiersz podsumowania
    # ------------------------------------------------------
    summary_rows.append(
        {
            "parameter": param,
            "eta_squared": eta_squared,
            "effect_range": effect_range,
            "spearman_corr": corr,
            "spearman_p_value": p_value,
            "best_value": best_value,
            "best_mean_accuracy": best_mean_accuracy,
            "worst_value": worst_value,
            "worst_mean_accuracy": worst_mean_accuracy,
        }
    )

# ==========================================================
# 3. TABELA PODSUMOWUJĄCA
# ==========================================================

summary = pd.DataFrame(summary_rows)
summary = summary.sort_values("eta_squared", ascending=False).reset_index(drop=True)

analysis_tables["summary"] = summary

# ==========================================================
# 4. ANALIZA INTERAKCJI PARAMETRÓW
# ==========================================================

# Usuwamy wiersze zawierające NaN lub inf, ponieważ anova_lm
# nie potrafi pracować z takimi wartościami.
df_interaction = df_mean.replace([np.inf, -np.inf], np.nan).dropna()

# Analiza interakcji może się nie udać, gdy:
# - danych jest zbyt mało,
# - nie wszystkie kombinacje parametrów występują,
# - macierz projektu jest osobliwa.
# W takim przypadku zapisujemy informację o błędzie do Excela.
try:
    interaction_model = ols(
        "accuracy ~ C(window_size) * C(epochs) * C(learning_rate)",
        data=df_interaction,
    ).fit()

    interaction_anova = sm.stats.anova_lm(interaction_model, typ=2)
    analysis_tables["interaction_anova"] = interaction_anova.reset_index()

except Exception as e:
    analysis_tables["interaction_anova"] = pd.DataFrame(
        {
            "message": [
                "Nie udało się obliczyć ANOVA interakcji."
            ],
            "details": [str(e)],
            "rows_used": [len(df_interaction)],
        }
    )

# ==========================================================
# 5. ZAPIS DO EXCEL
# ==========================================================

with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
    workbook = writer.book
    bold_format = workbook.add_format({"bold": True})

    # ------------------------------------------------------
    # Arkusze pivot (jak w Twoim oryginalnym kodzie)
    # ------------------------------------------------------
    for lr, table in pivot_tables.items():
        sheet_name = f"lr_{lr}"[:31]
        table.to_excel(writer, sheet_name=sheet_name)

        worksheet = writer.sheets[sheet_name]

        # maksimum w tabeli
        max_val = table.max().max()

        for r in range(table.shape[0]):
            for c in range(table.shape[1]):
                val = table.iloc[r, c]

                if pd.isna(val):
                    continue

                row = r + 1
                col = c + 1

                if val == max_val:
                    worksheet.write(row, col, val, bold_format)

    # ------------------------------------------------------
    # Arkusze analityczne
    # ------------------------------------------------------
    for sheet_name, table in analysis_tables.items():
        safe_name = sheet_name[:31]
        table.to_excel(writer, sheet_name=safe_name, index=False)

print("Zapisano:", OUTPUT_FILE)

# ==========================================================
# 6. WYDRUK PODSUMOWANIA
# ==========================================================

print("\n=== RANKING PARAMETRÓW WG ETA SQUARED ===")
print(
    summary[
        [
            "parameter",
            "eta_squared",
            "effect_range",
            "spearman_corr",
            "best_value",
            "best_mean_accuracy",
        ]
    ]
)
