from MatplotLibAPI import plot_line, plot_network, plot_table, plot_bubble, plot_pivotbar
import matplotlib.pyplot as plt
import pandas as pd


# Create a new figure
fig = plt.figure()
fig.patch.set_facecolor("black")
grid = plt.GridSpec(4, 2)


# Plot the time series on the new figure's axes
nw_df = pd.read_csv('nw.csv')
ax = fig.add_subplot(grid[0, 0:])
ax = plot_network(pd_df=nw_df, ax=ax)

ts_df = pd.read_csv('ts.csv')
ax2 = fig.add_subplot(grid[1, 0:])
ax2 = plot_line(
    pd_df=ts_df,
    label="dimension_type",
    x="dimension",
    y="segment_users",
    title="Campaign",
    ax=ax2
)
table_df = pd.read_csv('ts.csv')
ax3 = fig.add_subplot(grid[2, 0])
ax3 = plot_table(
    pd_df=ts_df,
    cols=["dimension_type", "dimension", "segment_users"],
    title="Top",
    ax=ax3,
    sort_by="segment_users",
    ascending=False,
    max_values=10
)
ax4 = fig.add_subplot(grid[2, 1])
ax4 = plot_table(
    pd_df=ts_df,
    cols=["dimension_type", "dimension", "segment_users"],
    title="Worst",
    ax=ax4,
    sort_by="segment_users",
    ascending=True,
    max_values=10
)
ax5 = fig.add_subplot(grid[3, 0:])
ax5 = plot_pivotbar(ts_df,
                    label="dimension_type",
                    x="dimension",
                    y="segment_users",
                    title='Example Pivot Bar')
fig.tight_layout()
plt.show()
