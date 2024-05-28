from MatplotLibAPI import plot_timeserie
import matplotlib.pyplot as plt
import pandas as pd




data=pd.read_csv('ts.csv')


# Create a new figure
fig = plt.figure(figsize=(19.2, 10.8), layout="tight")
fig.patch.set_facecolor("black")
ax = fig.add_subplot(1, 1, 1)

# Plot the time series on the new figure's axes
ax = plot_timeserie(
    pd_df=data,
    label="dimension_type",
    x="dimension",
    y="segment_users",
    title="Campaign",
    ax=ax
)

# Show the plot
plt.show()