import matplotlib.pyplot as plt
import pandas as pd
from anndata import AnnData




def stacked_bar(adata: AnnData,
                x: str,
                value: str,
                figsize: tuple = (8, 6),
                bar_width: float = 0.8,
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
    
    # Group and calculate percentages
    df = adata.obs.groupby([x, value]).size().unstack(fill_value=0)
    df_perc = df.div(df.sum(axis=1), axis=0) * 100

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
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
    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig, ax
