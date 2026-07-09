import pandas as pd
from pathlib import Path

INPUT_FILE = Path("evaluation/llm_f1_results.xlsx")
OUTPUT_DIR = Path("evaluation/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(INPUT_FILE)

required_cols = {"Model", "UseCase", "Domain", "F1"}
missing = required_cols - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")


def compute_borda(data: pd.DataFrame, scope_name: str):
    data = data.copy()

    # Rank models inside each use case
    data["Rank"] = data.groupby("UseCase")["F1"].rank(
        method="average",
        ascending=False
    )

    # Borda points: rank 1 gets N-1 points
    data["N_Models"] = data.groupby("UseCase")["Model"].transform("nunique")
    data["Borda_Points"] = data["N_Models"] - data["Rank"]

    summary = (
        data.groupby("Model")
        .agg(
            Mean_F1=("F1", "mean"),
            Total_Borda=("Borda_Points", "sum"),
            Mean_Borda=("Borda_Points", "mean")
        )
        .reset_index()
        .sort_values(["Total_Borda", "Mean_F1"], ascending=False)
    )

    data.to_csv(OUTPUT_DIR / f"borda_details_{scope_name}.csv", index=False)
    summary.to_csv(OUTPUT_DIR / f"borda_ranking_{scope_name}.csv", index=False)

    print(f"\n=== Borda Ranking: {scope_name} ===")
    print(summary)


def prepare_scott_knott(data: pd.DataFrame, scope_name: str):
    wide = (
        data.pivot_table(
            index="UseCase",
            columns="Model",
            values="F1",
            aggfunc="mean"
        )
        .reset_index()
    )

    output_file = OUTPUT_DIR / f"scott_knott_F1_{scope_name}_wide.csv"
    wide.to_csv(output_file, index=False)

    print(f"\nScott-Knott input created: {output_file}")


# Global analysis
compute_borda(df, "All")
prepare_scott_knott(df, "All")

# Per-domain analysis
for domain in df["Domain"].unique():
    domain_df = df[df["Domain"] == domain]
    safe_domain = domain.replace(" ", "_").replace("-", "_")

    compute_borda(domain_df, safe_domain)
    prepare_scott_knott(domain_df, safe_domain)