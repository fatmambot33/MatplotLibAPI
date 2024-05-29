# Hint for Visual Code Python Interactive window
# %%
from MatplotLibAPI import plot_timeserie, plot_network, plot_table, plot_pivotbar, plot_composite_bubble, MatPlotLibAccessor
import matplotlib.pyplot as plt
import pandas as pd


# bb_df = pd.read_csv('ts.csv')
# fig = bb_df.mpl.plot_bubble_composite(label="dimension",
#                                       x="index",
#                                       y="overlap",
#                                       z="users",
#                                       sort_by="index")
# fig.show()


nw_df = pd.read_csv('nw.csv')
ts_df = pd.read_csv('ts.csv')

fig, ax = plt.subplots()
ax = plot_network(pd_df=nw_df,
                  title='Network',
                  ax=ax)
fig.show()


fig, ax1 = plt.subplots()
ax1 = plot_composite_bubble(
    pd_df=ts_df,
    label="dimension",
    x="index",
    y="overlap",
    z="users",
    title='Example TimeSerie'
)
fig.show()


fig, ax2 = plt.subplots()
ax2 = plot_timeserie(
    pd_df=ts_df,
    label="dimension_type",
    x="dimension",
    y="segment_users",
    title='Example TimeSerie',
    ax=ax2
)
fig.show()


fig, ax3 = plt.subplots()
ax3 = plot_table(
    pd_df=ts_df,
    cols=["dimension_type", "dimension", "segment_users"],
    title='Example Top Table',
    ax=ax3,
    sort_by="segment_users",
    ascending=False,
    max_values=10
)
fig.show()


fig, ax4 = plt.subplots()
ax4 = plot_table(
    pd_df=ts_df,
    cols=["dimension_type", "dimension", "segment_users"],
    title='Example Worst Table',
    ax=ax4,
    sort_by="segment_users",
    ascending=True,
    max_values=10
)
fig.show()

fig, ax5 = plt.subplots()
ax5 = plot_pivotbar(ts_df,
                    label="dimension_type",
                    x="dimension",
                    y="segment_users",
                    title='Example Pivot Bar')

fig.show()


# %%
