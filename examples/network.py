import sys
import os
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))



def generate_sample_network_data():
    """Generate and save sample data for a network graph."""
    data = {
        "city_a": ["New York", "London", "Tokyo", "Sydney", "New York"],
        "city_b": ["London", "Tokyo", "Sydney", "New York", "Tokyo"],
        "distance_km": [5585, 9562, 7824, 16027, 10850],
    }
    df = pd.DataFrame(data)
    df.to_csv("data/network.csv", index=False)



def plot_sample_network_data():
    """Load a sample DataFrame for testing."""
    
    from MatplotLibAPI.network import fplot_network, NetworkGraph
    # Assuming sample data is stored in a 'data' directory within the tests folder
    filepath = os.path.join("data", "network.csv")
    pd_df = pd.read_csv(filepath)
    graph = NetworkGraph.from_pandas_edgelist(
        pd_df, source="city_a", target="city_b", edge_weight_col="distance_km"
    )
    plot_fig=graph.fplot_network(title="Network Graph", edge_weight_col="distance_km")
   
    return plot_fig

if __name__ == "__main__":
    generate_sample_network_data()
    plot_fig = plot_sample_network_data()
    fig_path = os.path.join("data", "network.png")
    plot_fig.savefig(fig_path)
    #plot_fig.show()