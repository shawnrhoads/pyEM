import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_choices(choices_A, filename=None):
    """
    Plots the subject average EV over trials

    Inputs:
        - choices_A (np.array): subject choices for option A
        - filename (str): filename to save figure to (if None, figure is not saved)
    """
    nsubjects, nblocks, ntrials = choices_A.shape
    choices_B = np.ones_like(choices_A) - choices_A
    df_a = pd.DataFrame(np.mean(choices_A, axis=1), 
                        columns=[f'trial{i}' for i in range(ntrials)]).rename_axis('subject', axis=0).reset_index()
    df_b = pd.DataFrame(np.mean(choices_B, axis=1), 
                        columns=[f'trial{i}' for i in range(ntrials)]).rename_axis('subject', axis=0).reset_index()
    c_longA = pd.wide_to_long(df_a, stubnames='trial', i='subject', j='choices')
    c_longB = pd.wide_to_long(df_b, stubnames='trial', i='subject', j='choices')

    sns.lineplot(x='choices', y='trial',data=c_longA, label='A')
    sns.lineplot(x='choices', y='trial',data=c_longB, label='B')
    sns.despine()
    plt.xlabel('Trials')
    plt.ylabel('Choice Frequency')
    plt.legend()

    if filename:
        plt.savefig(filename, dpi=450, bbox_inches='tight')

    plt.show()

def plot_scatter(
    x,
    xlabel,
    y,
    ylabel,
    *,
    ax=None,
    show_line=True,
    equal_limits=True,
    s=75,
    alpha=0.6,
    colorname='royalblue',
    annotate=True,
):
    """
    Scatter plot of ``x`` vs ``y`` with optional Pearson r annotation and
    x=y reference line.  The function deliberately does **not** call
    ``plt.show()`` so that callers can aggregate multiple subplots before
    displaying the figure.

    Args:
        x (array-like): x-axis data
        xlabel (str): x-axis label
        y (array-like): y-axis data
        ylabel (str): y-axis label
        ax (matplotlib.axes.Axes | None): Axes to draw on. If None, creates a new figure/axes.
        show_line (bool): Whether to draw dashed x=y line.
        equal_limits (bool): If True, sets identical limits and equal aspect.
        s (float): Marker size.
        alpha (float): Marker opacity.
        colorname (str): Marker color.
        annotate (bool): If True, adds Pearson r annotation.

    Returns:
        matplotlib.axes.Axes: The axes the plot was drawn on.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    created_fig = None
    if ax is None:
        # If used standalone, give it a sensible size and turn on constrained layout
        created_fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.6), constrained_layout=True)

    # Scatter
    ax.scatter(x, y, s=s, alpha=alpha, color=colorname)

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Pearson r annotation
    if annotate:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any() and mask.sum() > 1:
            corr = np.corrcoef(x[mask], y[mask])[0, 1]
            ax.annotate(f'Pearson r = {corr:.2f}', xy=(0.05, 0.95),
                        xycoords='axes fraction', va='top', fontsize=11)

    # Optional x=y line and equal limits
    if show_line:
        data_min = np.nanmin([np.nanmin(x, initial=np.nan), np.nanmin(y, initial=np.nan)])
        data_max = np.nanmax([np.nanmax(x, initial=np.nan), np.nanmax(y, initial=np.nan)])
        cur_xmin, cur_xmax = ax.get_xlim()
        cur_ymin, cur_ymax = ax.get_ylim()
        lo = np.nanmin([data_min, cur_xmin, cur_ymin])
        hi = np.nanmax([data_max, cur_xmax, cur_ymax])
        ax.plot([lo, hi], [lo, hi], linestyle='--', color='k', alpha=0.6, zorder=0)

        if equal_limits:
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            # Keep the box square without triggering excessive outer padding
            try:
                ax.set_box_aspect(1)
            except Exception:
                pass

    # A touch of margin so points/labels aren’t flush against the frame
    ax.margins(0.05)

    sns.despine()
    return ax

