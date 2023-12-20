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

def plot_scatter(x, xlabel, y, ylabel, filename=None):
    """
    Plots a scatterplot of x vs y with a Pearson correlation coefficient in the top left corner

    Inputs:
        - `x` (np.array): x-axis data
        - `xlabel` (str): x-axis label
        - `y` (np.array): y-axis data
        - `ylabel` (str): y-axis label
    """
    df = pd.DataFrame({xlabel:x, ylabel:y})

    # Plot a pointplot of Mean Loneliness as a function of Depression
    ax = sns.scatterplot(x=xlabel, 
                        y=ylabel,
                        s=75,            # set size of points to 50
                        alpha=0.25,      # set opacity to 0.15
                        color='royalblue',  # set color
                        data=df)

    # Compute the correlation between Loneliness and Depression
    corr = df[xlabel].corr(df[ylabel], method='pearson')

    # Annotation with the correlation in the top left corner with small font size
    ax.annotate(f'Pearson r = {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)

    # Set the axis ranges
    if 'beta' in xlabel:
        ax.set_xlim([0, 10.05])
        ax.set_ylim([0, 10.05])
    elif 'lr' in xlabel:
        ax.set_xlim([-.02, 1.02])
        ax.set_ylim([-.02, 1.02])

    # we can also tidy up some more by removing the top and right spines
    sns.despine()

    if filename:
        plt.savefig(filename, dpi=450, bbox_inches='tight')

    plt.show()