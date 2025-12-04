"""Generate sample data for MatplotLibAPI examples."""

import os
import pandas as pd


def generate_bubble_data():
    """Generate and save sample data for a bubble chart."""
    data = {
        "country": ["USA", "China", "India", "Brazil", "Nigeria"],
        "population": [331, 1441, 1393, 213, 206],  # in millions
        "gdp_per_capita": [63593, 10500, 2191, 7741, 2229],
        "continent": ["North America", "Asia", "Asia", "South America", "Africa"],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/bubble.csv", index=False)


def generate_network_data():
    """Generate and save sample data for a network graph."""
    data = {
        "city_a": ["New York", "London", "Tokyo", "Sydney", "New York"],
        "city_b": ["London", "Tokyo", "Sydney", "New York", "Tokyo"],
        "distance_km": [5585, 9562, 7824, 16027, 10850],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/network.csv", index=False)


def generate_pivot_data():
    """Generate and save sample data for a pivot table."""
    data = {
        "year": [2020, 2020, 2021, 2021, 2022, 2022],
        "city": ["New York", "Tokyo", "New York", "Tokyo", "New York", "Tokyo"],
        "population_increase": [10000, 5000, 12000, 4000, 11000, 6000],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/pivot.csv", index=False)


def generate_table_data():
    """Generate and save sample data for a table."""
    data = {
        "country": ["USA", "China", "India", "Brazil", "Nigeria"],
        "capital": ["Washington D.C.", "Beijing", "New Delhi", "Brasília", "Abuja"],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/table.csv", index=False)


def generate_timeserie_data():
    """Generate and save sample data for a timeseries plot."""
    data = {
        "year": pd.to_datetime(
            [
                "2020-01-01",
                "2021-01-01",
                "2022-01-01",
                "2020-01-01",
                "2021-01-01",
                "2022-01-01",
            ]
        ),
        "city": ["New York", "New York", "New York", "Tokyo", "Tokyo", "Tokyo"],
        "population": [8.4, 8.5, 8.6, 13.9, 13.9, 14.0],  # in millions
    }
    df = pd.DataFrame(data)
    df.to_csv("data/timeserie.csv", index=False)


def generate_treemap_data():
    """Generate and save sample data for a treemap."""
    data = {
        "location": [
            "North America",
            "Asia",
            "South America",
            "Africa",
            "USA",
            "China",
            "India",
            "Brazil",
            "Nigeria",
        ],
        "parent": [
            "",
            "",
            "",
            "",
            "North America",
            "Asia",
            "Asia",
            "South America",
            "Africa",
        ],
        "population": [579, 4561, 422, 1216, 331, 1441, 1393, 213, 206],  # in millions
    }
    df = pd.DataFrame(data)
    # For treemap, we need a path-like structure. We will create it here.
    df["path"] = df["parent"] + "/" + df["location"]
    df["path"] = df["path"].str.lstrip("/")
    df.to_csv("data/treemap.csv", index=False)


def generate_sunburst_data():
    """Generate and save sample data for a sunburst chart."""
    data = {
        "name": [
            "World",
            "North America",
            "Asia",
            "South America",
            "Africa",
            "USA",
            "China",
            "India",
            "Brazil",
            "Nigeria",
        ],
        "parent": [
            "",
            "World",
            "World",
            "World",
            "World",
            "North America",
            "Asia",
            "Asia",
            "South America",
            "Africa",
        ],
        "population": [
            7000,
            579,
            4561,
            422,
            1216,
            331,
            1441,
            1393,
            213,
            206,
        ],  # in millions
    }
    df = pd.DataFrame(data)
    df.to_csv("data/sunburst.csv", index=False)


def generate_wordcloud_data():
    """Generate and save sample data for a word cloud."""
    data = {
        "country": [
            "USA",
            "China",
            "India",
            "Brazil",
            "Nigeria",
            "Russia",
            "Japan",
            "Germany",
            "UK",
            "France",
        ],
        "population": [331, 1441, 1393, 213, 206, 145, 126, 83, 67, 65],  # in millions
    }
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
