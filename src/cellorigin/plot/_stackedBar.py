import matplotlib.pyplot as plt
import pandas as pd
from anndata import AnnData





def stacked_bar(adata: AnnData,
                x: str,
                value: str,
                figsize: tuple = (8, 6),
                bar_width: float = 0.8,
                cmap: str | dict = None, 
                legend_kwargs: dict = None,
                return_fig: bool = False,
                save: str = None,
                show: bool = True):
    
    """
    Create a stacked bar plot from an AnnData object's obs DataFrame.
    
    Parameters:
    ----------
    adata : AnnData
        AnnData object with observations to group and plot.
    x : str
        Column name in adata.obs for the x-axis categories.
    value : str
        Column name in adata.obs for the values to stack by.
    figsize : tuple, optional
        Figure size, default is (8, 6).
    bar_width : float, optional
        Width of the bars, default is 0.8.
    cmap : str or dict
        - If `str`: Matplotlib colormap name (e.g., "viridis", "tab10").
        - If `dict`: Dictionary mapping `value` categories to specific colors.
    legend_kwargs : dict, optional
        Additional keyword arguments for the legend.
    return_fig : bool, optional
        Whether to return the figure and axis objects, default is False.
    save : str, optional
        Filepath to save the figure, default is None.
    show : bool, optional
        Whether to display the plot, default is True.
        
    Returns:
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes._axes.Axes
        If return_fig is True, returns the figure and axis objects.
    """

    # Validate input
    if x not in adata.obs.columns or value not in adata.obs.columns:
        raise ValueError(f"Columns '{x}' or '{value}' not found in column names of adata.obs")
    
    unique_values = list(adata.obs[value].unique())  # Use list to maintain order

    # Group and calculate percentages
    df = adata.obs.groupby([x, value], observed=False).size().unstack(fill_value=0)
    df_perc = df.div(df.sum(axis=1).values[:, None], axis=0) * 100  # Prevents FutureWarning

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Handle colormap input
    if isinstance(cmap, str):  # If cmap is a colormap name, use Matplotlib's colormap
        df_perc.plot(kind='bar', stacked=True, ax=ax, width=bar_width, colormap=cmap)
    
    elif isinstance(cmap, dict):  # If cmap is a dictionary, use custom colors
        # Ensure all unique values exist in cmap
        missing_values = [v for v in unique_values if v not in cmap]
        
        if missing_values:
            raise ValueError(f"Missing colors for values: {missing_values}. Ensure all values in `{value}` have assigned colors.")

        # Reorder DataFrame columns to match dictionary order
        df_perc = df_perc[list(cmap.keys())]

        # Plot with dictionary-based color mapping
        df_perc.plot(kind='bar', stacked=True, ax=ax, width=bar_width, color=list(cmap.values()))

    else:
        df_perc.plot(kind='bar', stacked=True, ax=ax, width=bar_width)

    # Set labels
    ax.set_ylabel('Percentage (%)', fontsize=20, weight="bold")
    ax.set_xlabel(f'{x}', fontsize=20, weight="bold")

    # Customize legend
    if legend_kwargs is None:
        legend_kwargs = {}
    ax.legend(**legend_kwargs)

    # Handle output
    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
    
    if return_fig:
        return fig, ax  # Let user handle show()
    
    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)
