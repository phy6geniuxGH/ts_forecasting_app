import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import datetime
from tabulate import tabulate

def ts_plotter(
    data: pd.DataFrame,
    title: str = "Time Series Data",
    x_label: str = "Date",
    y_label: str = "Value",
    plot_method: str = "matplotlib",
    set_path: str = "./saved_figures/"
) -> None:
    """
    Function that handles plotting of time series data using specified plotting library (Matplotlib/Seaborn or Plotly).

    Parameters:
    - data (pd.DataFrame): DataFrame containing 'ds' (date) and 'y' (value) columns.
    - title (str): Title of the plot.
    - x_label (str): Label for the x-axis.
    - y_label (str): Label for the y-axis.
    - plot_method (str): Method to use for plotting ('matplotlib', 'seaborn', or 'plotly').

    Returns:
    - None
    """
    # Validate input data
    if 'ds' not in data.columns or 'y' not in data.columns:
        raise ValueError("DataFrame must contain 'ds' and 'y' columns.")
    else:
        data = data.sort_values(by='ds')
        print(f"Data contains {len(data)} records from {data['ds'].min()} to {data['ds'].max()}.")
        print(tabulate(data.head(), headers='keys', tablefmt='psql'))
    # Proceed to data plotting
    try:
        if plot_method == "plotly":
            fig = plotly.express.line(data, x='ds', y='y', title=title, labels={ 'ds': x_label, 'y': y_label })
            fig.show()
        elif plot_method == "seaborn":
            plt.figure(figsize=(15, 8))
            sns.set_style("darkgrid")
            sns.lineplot(data=data, x='ds', y='y')
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{set_path}time_series_plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.show()
    except Exception as e:
        print(f"Error plotting data: {e}")

# Usage example
if __name__ == "__main__":
    # Generate some example data
    date_range = pd.date_range(start="2023-01-01", periods=100, freq='D')
    values = np.random.rand(100).cumsum()
    example_data = pd.DataFrame({'ds': date_range, 'y': values})
    
    # Plot using the function
    ts_plotter(example_data, title="Example Time Series Plot", plot_method="seaborn")