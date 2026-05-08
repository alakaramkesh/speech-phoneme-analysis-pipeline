import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PARAMS_PATH = os.path.join(BASE_DIR, "params.yaml")

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)

INPUT_CSV = os.path.join(BASE_DIR, params["plots"]["input_csv"])

VOWEL_CHART_OUTPUT = os.path.join(
    BASE_DIR,
    params["plots"]["vowel_chart_output"]
)

BOXPLOT_OUTPUT = os.path.join(
    BASE_DIR,
    params["plots"]["boxplot_output"]
)
VIOLIN_PLOT_OUTPUT = os.path.join(
    BASE_DIR,
    params["plots"]["violin_plot_output"]
)

# Load normalized acoustic features
# We load the dataframe once and reuse it for all plots
DF = pd.read_csv(INPUT_CSV)


# Create combined speaker group labels
DF["group"] = DF["L1"] + "/" + DF["gender"]
GROUP_ORDER = ["L1/f", "L1/m", "L2/f", "L2/m"]


# Create figure output directory
os.makedirs(os.path.dirname(VOWEL_CHART_OUTPUT), exist_ok=True)



def plot_vowel_chart(df):

    # Compute vowel centroids
    centroids = df.groupby(["label", "group"])[["F1_norm", "F2_norm"]].mean().reset_index()

    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        data=centroids,
        x="F2_norm",
        y="F1_norm",
        hue="group",
        style="group",
        hue_order=GROUP_ORDER,
        style_order=GROUP_ORDER,
        s=150
    )

    # Add vowel labels
    for _, row in centroids.iterrows():
        plt.text(
            row["F2_norm"] + 0.02,
            row["F1_norm"] + 0.02,
            row["label"],
            fontsize=10
        )

    # Invert axes following IPA convention
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()

    plt.title("French vowel space after Lobanov normalization")
    plt.xlabel("F2 normalized")
    plt.ylabel("F1 normalized")

    plt.legend(title="Speaker group")

    plt.tight_layout()

    plt.savefig(VOWEL_CHART_OUTPUT, dpi=300)

    print(f"Saved vowel chart to: {VOWEL_CHART_OUTPUT}")

    plt.close()


def plot_boxplots(df):

    # Create two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # F1 boxplot
    sns.boxplot(
        data=df,
        x="label",
        y="F1_norm",
        hue="group",
        hue_order=GROUP_ORDER,
        ax=axes[0],
        showfliers=False
    )

    axes[0].set_title("F1 distribution by phoneme and speaker group")
    axes[0].set_xlabel("Phoneme")
    axes[0].set_ylabel("Normalized F1")

    # F2 boxplot
    sns.boxplot(
        data=df,
        x="label",
        y="F2_norm",
        hue="group",
        hue_order=GROUP_ORDER,
        ax=axes[1],
        showfliers=False
    )

    axes[1].set_title("F2 distribution by phoneme and speaker group")
    axes[1].set_xlabel("Phoneme")
    axes[1].set_ylabel("Normalized F2")

    plt.tight_layout()

    plt.savefig(BOXPLOT_OUTPUT, dpi=300)

    print(f"Saved boxplots to: {BOXPLOT_OUTPUT}")

    plt.close()

def plot_violin_plots(df):
    # Select a subset of vowels
    selected_vowels = ["i", "a", "u"]
    subset_df = df[df["label"].isin(selected_vowels)]
    plt.figure(figsize=(12, 8))
    # Violin plot
    sns.violinplot(
        data=subset_df,
        x="label",
        y="F1_norm",
        hue="group",
        hue_order=GROUP_ORDER,
        inner=None
    )
    # Overlay individual points
    sns.stripplot(
        data=subset_df,
        x="label",
        y="F1_norm",
        hue="group",
        hue_order=GROUP_ORDER,
        dodge=True,
        alpha=0.4,
        size=3
    )
    plt.title("Intra-speaker variability across repetitions")
    plt.xlabel("Phoneme")
    plt.ylabel("Normalized F1")
    plt.tight_layout()
    plt.savefig(VIOLIN_PLOT_OUTPUT, dpi=300)
    print(f"Saved violin plots to: {VIOLIN_PLOT_OUTPUT}")
    plt.close()

def main():
    plot_vowel_chart(DF)
    plot_boxplots(DF)
    plot_violin_plots(DF)


if __name__ == "__main__":
    main()