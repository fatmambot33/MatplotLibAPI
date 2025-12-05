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


def generate_bar_data():
    """Generate and save sample data for bar and stacked bar charts."""

    data = {
        "product": ["Gadget", "Gadget", "Gadget", "Widget", "Widget", "Widget"],
        "region": ["North", "South", "West", "North", "South", "West"],
        "revenue": [120000, 95000, 88000, 110000, 102000, 76000],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/bar.csv", index=False)


def generate_histogram_data():
    """Generate and save sample data for histogram and KDE plots."""

    data = {
        "waiting_time_minutes": [
            5,
            7,
            6,
            12,
            15,
            9,
            4,
            18,
            20,
            11,
            6,
            7,
            9,
            14,
            16,
            8,
            10,
            6,
            13,
            12,
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv("data/histogram.csv", index=False)


def generate_box_violin_data():
    """Generate and save sample data for box and violin plots."""

    data = {
        "department": [
            "Engineering",
            "Engineering",
            "Engineering",
            "Sales",
            "Sales",
            "Sales",
            "Support",
            "Support",
            "Support",
        ],
        "satisfaction_score": [
            7.8,
            8.3,
            7.5,
            6.4,
            6.9,
            7.2,
            7.0,
            7.4,
            6.8,
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/box_violin.csv", index=False)


def generate_heatmap_and_correlation_data():
    """Generate and save sample data for heatmaps and correlation matrices."""

    data = {
        "month": ["Jan", "Jan", "Feb", "Feb", "Mar", "Mar", "Apr", "Apr"],
        "channel": ["Email", "Social", "Email", "Social", "Email", "Social", "Email", "Social"],
        "engagements": [340, 420, 380, 510, 410, 560, 390, 530],
        "conversions": [34, 28, 30, 32, 36, 35, 33, 31],
        "cost_per_click": [0.75, 0.65, 0.72, 0.68, 0.74, 0.64, 0.73, 0.66],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/heatmap.csv", index=False)
    df[["engagements", "conversions", "cost_per_click"]].to_csv(
        "data/correlation.csv", index=False
    )


def generate_area_data():
    """Generate and save sample data for area charts."""

    data = {
        "quarter": ["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"],
        "segment": ["SMB", "SMB", "SMB", "SMB", "Enterprise", "Enterprise", "Enterprise", "Enterprise"],
        "subscriptions": [120, 150, 170, 190, 200, 230, 260, 300],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/area.csv", index=False)


def generate_pie_waffle_data():
    """Generate and save sample data for pie, donut, and waffle charts."""

    data = {
        "device": ["Desktop", "Mobile", "Tablet", "Other"],
        "sessions": [5200, 8900, 1300, 600],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/pie.csv", index=False)
    df.to_csv("data/waffle.csv", index=False)


def generate_sankey_data():
    """Generate and save sample data for Sankey diagrams."""

    data = {
        "source": ["Homepage", "Homepage", "Landing Page", "Landing Page", "Cart"],
        "target": ["Landing Page", "Product", "Product", "Signup", "Checkout"],
        "value": [3000, 1500, 1200, 800, 500],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/sankey.csv", index=False)


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
    generate_bar_data()
    generate_histogram_data()
    generate_box_violin_data()
    generate_heatmap_and_correlation_data()
    generate_area_data()
    generate_pie_waffle_data()
    generate_sankey_data()


if __name__ == "__main__":
    main()
