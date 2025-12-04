"""Generate sample data for MatplotLibAPI examples."""

import os
import pandas as pd


def generate_bubble_data():
    """Generate and save sample data for a bubble chart."""
    data = {
        "country": ["A", "B", "C", "D"],
        "gdp_per_capita": [45000, 42000, 52000, 48000],
        "life_expectancy": [81, 78, 83, 82],
        "population": [10, 20, 5, 30],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/bubble.csv", index=False)


def generate_network_data():
    """Generate and save sample data for a network graph."""
    data = {
        "source": ["A", "B", "C", "D"],
        "target": ["B", "C", "D", "A"],
        "weight": [1, 1, 1, 1],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/network.csv", index=False)


def generate_pivot_data():
    """Generate and save sample data for a pivot table."""
    data = {
        "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
        "category": ["A", "B", "A", "B"],
        "value": [10, 20, 15, 25],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/pivot.csv", index=False)


def generate_table_data():
    """Generate and save sample data for a table."""
    data = {"col1": [1, 2, 3], "col2": ["A", "B", "C"]}
    df = pd.DataFrame(data)
    df.to_csv("data/table.csv", index=False)


def generate_timeserie_data():
    """Generate and save sample data for a timeseries plot."""
    data = {
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "group": ["A", "A", "B"],
        "value": [1, 2, 3],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/timeserie.csv", index=False)


def generate_treemap_data():
    """Generate and save sample data for a treemap."""
    data = {"path": ["A", "B", "C"], "values": [10, 20, 30]}
    df = pd.DataFrame(data)
    df.to_csv("data/treemap.csv", index=False)


def generate_sunburst_data():
    """Generate and save sample data for a sunburst chart."""
    data = {
        "labels": [
            "Eve",
            "Cain",
            "Seth",
            "Enos",
            "Noam",
            "Abel",
            "Awan",
            "Enoch",
            "Azura",
        ],
        "parents": ["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve"],
        "values": [10, 14, 12, 10, 2, 6, 6, 4, 4],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/sunburst.csv", index=False)


def generate_wordcloud_data():
    """Generate and save sample data for a word cloud."""
    data = {"word": ["alpha", "beta", "gamma", "alpha"], "weight": [2, 1, 3, 1]}
    df = pd.DataFrame(data)
    df.to_csv("data/wordcloud.csv", index=False)


def main():
    """Generate all sample data."""
    if not os.path.exists("data"):
        os.makedirs("data")

    generate_bubble_data()
    generate_network_data()
    generate_pivot_data()
    generate_table_data()
    generate_timeserie_data()
    generate_treemap_data()
    generate_sunburst_data()
    generate_wordcloud_data()


if __name__ == "__main__":
    main()
