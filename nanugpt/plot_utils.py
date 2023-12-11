os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # needed to avoid Jupyter kernal crash due to matplotlib
import matplotlib
# below is needed to avoid message ""Backend TkAgg is interactive backend. Turning interactive mode on"
matplotlib.use('TkAgg',force=True)
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def draw_histogram(data, xlabel='Values', ylabel='Frequency', title='Histogram', bins=None, log_x=False, log_y=False):
    unique_vals = np.unique(data)
    if len(unique_vals) < 10:
        bins = len(unique_vals)  # If the unique values are less than 10, make a bin for each
    elif bins is None:  # Automatic bin sizing using Freedman-Diaconis rule
        q75, q25 = np.percentile(data, [75 ,25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(data) ** (1/3))
        bins = min(round((np.max(data) - np.min(data)) / bin_width), 1000)

    n, bins, patches = plt.hist(data, bins=bins, edgecolor='black')

    # Create a normalization object which scales data values to the range [0, 1]
    fracs = n / n.max()
    norm = mcolors.Normalize(fracs.min(), fracs.max())

    # Assigning a color for each bar using the 'viridis' colormap
    for thisfrac, thispatch in zip(fracs, patches):
        color = cm.viridis(norm(thisfrac)) # type: ignore
        thispatch.set_facecolor(color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')

    plt.show()
