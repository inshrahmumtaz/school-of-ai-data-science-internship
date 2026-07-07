import matplotlib.pyplot as plt


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        f"../reports/figures/{filename}",
        dpi=200,
        bbox_inches="tight"
    )


def format_currency(value):
    return f"${value:,.2f}"


def describe_missingness(df):
    missing = df.isnull().sum()

    return missing[missing > 0]