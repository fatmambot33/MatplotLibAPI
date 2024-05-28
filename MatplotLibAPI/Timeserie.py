
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from .Utils import (TIMESERIE_STYLE_TEMPLATE, DynamicFuncFormatter,
                    StyleTemplate,string_formatter)

# region TimeSeries


# def plot_timeseries(ax: Axes,
#                     data: pd.DataFrame,
#                     x_col: str,
#                     y_col: Union[str, List[str]],
#                     fig_title:  Optional[str] = None,
#                     style:  Optional[StyleTemplate] = None,

#                     rolling_days: int = 30,
#                     highlight: Optional[List[datetime]] = None,
#                     **kwargs) -> Axes:
#     """
#     Plots a time series with the actual data, rolling mean, and standard deviation of multiple metrics.
#     Highlights the specified dates and the dates with the 5 highest cumulative sum of residuals
#     for each metric, ensuring each date is at least 10% of the total timeframe apart from the others.

#     Parameters:
#         data (pd.DataFrame): Data to plot. Must have a datetime index.
#         metrics (List[str]): List of column names in 'data' to plot.
#         rolling_days (int, optional): Window size for the rolling mean and standard deviation. Default is 30.
#         title (str, optional): Title for the plot. If None, no title is set. Default is None.
#         highlight (List[datetime], optional): List of dates to highlight. If None, highlights the entire timeframe. 
#             Default is None.

#     Returns:
#         plt.Figure: The created matplotlib Figure object.
#     """
#     if type(y_col) == str:
#         y_col = [y_col]
#     # Clear the axis before plotting
#     ax.clear()
#     if fig_title is not None:
#         ax.set_title(fig_title)
#     if style is None:
#         style = PIVOTLINES_STYLE_TEMPLATE
#     ax.figure.set_facecolor(style.background_color)
#     ax.figure.set_edgecolor(style.fig_border)
#     if x_col in data.columns:
#         data[x_col] = pd.to_datetime(data[x_col])
#         data.sort_values(by=x_col)
#         data.set_index(x_col, inplace=True)
#     total_days = (data.index.max() - data.index.min()).days
#     min_interval = total_days * 0.1

#     for metric in y_col:
#         # Compute rolling mean, standard deviation, residuals, and cusum for each metric
#         rolling_mean = data[metric].rolling(window=rolling_days).mean()
#         rolling_std = data[metric].rolling(window=rolling_days).std()
#         residuals = data[metric] - rolling_mean
#         residuals.dropna(inplace=True)
#         cusum = np.cumsum(residuals)
#         # Prepare dates to highlight
#         if highlight is None:
#             highlight = [data.index.min(), data.index.max()]
#             z_scores = (data[metric] - rolling_mean) / rolling_std
#             z_scores.dropna(inplace=True)
#             Q1 = z_scores.quantile(0.25)
#             Q3 = z_scores.quantile(0.75)
#             IQR = Q3 - Q1
#             z_threshold = Q3 + 1.5 * IQR
#             # Find dates with z-scores above the threshold
#             outlier_dates = z_scores[z_scores > z_threshold].index.tolist()

#             sorted_z_scores = z_scores.sort_values(ascending=False)
#             top_5_z_dates = []
#             for date, value in sorted_z_scores.items():
#                 if value >= z_threshold:
#                     if all(abs((date - d).days) >= min_interval for d in top_5_z_dates):
#                         top_5_z_dates.append(date)
#                         if len(top_5_z_dates) >= 5:
#                             break
#             highlight += top_5_z_dates
#         else:
#             highlight.sort()

#             min_date = data.index.min()
#             if min_date < highlight[0]:
#                 highlight.insert(0, min_date)

#             max_date = data.index.max()
#             if max_date < highlight[len(highlight)-1]:
#                 highlight.append(max_date)

#         # Plot the metric, its rolling mean, and standard deviation
#         # Get the line object to extract the color
#         line, = ax.plot(data[metric], label=metric)
#         ax.plot(rolling_mean, color=line.get_color(), linewidth=line.get_linewidth()
#                 * 3, label='_nolegend_')  # Use the same color for the rolling mean
#         ax.fill_between(rolling_std.index,
#                         rolling_mean - rolling_std,
#                         rolling_mean + rolling_std,
#                         alpha=0.2)

#     # Sort and deduplicate the highlight dates
#     highlight = sorted(set(highlight))

#     # Calculate mean of each metric between each pair of consecutive highlight dates
#     for i in range(len(highlight) - 1):
#         start_date = highlight[i]
#         end_date = highlight[i+1]
#         for metric in y_col:
#             metric_mean = data.loc[start_date:end_date, metric].mean()
#             ax.hlines(y=metric_mean, xmin=start_date, xmax=end_date,
#                       linestyle='--', color=style.font_color, alpha=0.5)
#             # ax.text(start_date, metric_mean, start_date.strftime('%Y-%m-%d'),
#             #        va='center', ha='right', backgroundcolor='w')

#         # Add vertical lines for highlight dates
#         ax.axvline(x=start_date, color=style.font_color, linestyle='--')

#     ax.xaxis.set_major_formatter(DynamicFuncFormatter(style.x_formatter))
#     ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#     if style.y_formatter is None:
#         ylabels = ['{:,.0f}%'.format(y) for y in ax.get_yticks()*100]
#         ax.set_yticklabels(ylabels)
#     else:
#         ax.yaxis.set_major_formatter(DynamicFuncFormatter(style.y_formatter))
#     if style.legend:
#         ax.legend(loc='best')
#     return ax

def plot_timeserie(pd_df: pd.DataFrame,
                   label: str,
                   x: str,
                   y: str,
                   title: str = "Test",
                   style:StyleTemplate=TIMESERIE_STYLE_TEMPLATE,
                     ax=None)->Axes:

    df = pd_df[[label, x, y]].sort_values(by=[label, x])
    df[x] = pd.to_datetime(df[x])
    df.set_index(x, inplace=True)

    sns.set_palette(style.palette)
    if ax is None:
        ax = plt.gca()
    ax.set_facecolor(style.background_color)
    # Get unique dimension_types
    label_types = df[label].unique()

    for label_type in label_types:
        temp_df = df[df[label] == label_type]
        temp_df = temp_df.sort_values(by=x) 
        if style.format_funcs.get("label"):
            label=style.format_funcs.get("label")(label_type)
        ax.plot(temp_df.index,
                temp_df[y],
                linestyle='-',
                label=label_type)

    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=style.font_size-2,
        title_fontsize=style.font_size+2,
        labelcolor='linecolor',
        facecolor=style.background_color)

    ax.set_xlabel(string_formatter(x), color=style.font_color)
    if style.format_funcs.get("x"):
        ax.xaxis.set_major_formatter(DynamicFuncFormatter(style.format_funcs.get("x")))
    ax.tick_params(axis='x', colors=style.font_color, labelrotation=45, labelsize=style.font_size-4)
    
    ax.set_ylabel(string_formatter(y), color=style.font_color)
    if style.format_funcs.get("y"):
        ax.yaxis.set_major_formatter(DynamicFuncFormatter(style.format_funcs.get("y")))
    ax.tick_params(axis='y', colors=style.font_color, labelsize=style.font_size-4)
    
    ax.grid(True)
    ax.set_title(title)
    return ax



# endregion


