import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class ColumnPlotter:
    r"""
    A very basic class to visually compare matrices by plotting
    columns in an interactive window.

    The user can scroll forward and backward through the columns of
    data, while using the plotting tools provided by `matplotlib
    <https://matplotlib.org>`_ (such as zoom).

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from pyyeti.column_plotter import ColumnPlotter
        >>> #
        >>> h = 0.01
        >>> t = np.arange(0, 1, h)
        >>> sin = np.sin(3 * 2 * np.pi * t)
        >>> sin_a = sin + np.random.randn(*sin.shape) * 0.1
        >>> sin_b = sin + np.random.randn(*sin.shape) * 0.1
        >>> A = np.column_stack((sin, sin_a, sin_b))
        >>> B = A + np.random.randn(*A.shape) * 0.2 + 2
        >>> C = A + np.random.randn(*A.shape) * 0.2 + 3
        >>> #
        >>> cp = ColumnPlotter(
        ...     t,
        ...     dict(A=A, B=B, C=C),
        ...     ["1st column", "2nd column", "3rd column"],
        ... )
    """

    def __init__(self, x, dct, column_labels=None):
        """
        Instantiates a :class:`ColumnPlotter` object

        Parameters
        ----------
        x : 1d array_like
            The x-axis data
        dct : dict
            Dictionary of y-axis data. The key is the label for the
            data and will be in the legend. The values are the
            matrices to compare and the number of rows must equal
            ``len(x)``. All matrices are expected to be the same size.
        column_labels : list or None; optional
            List of strings to be used for the plot titles;
            ``len(column_labels)`` is expected to be equal to the
            number of columns in the y-axis data matrices.
        """
        self.x = x
        self.dct = dct
        self.column_labels = column_labels
        self.ind = 0
        fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        self.lines = {}
        self.n_columns = None
        for key, value in self.dct.items():
            line = plt.plot(self.x, value[:, self.ind], label=key)[0]
            self.lines[line] = value
            if self.n_columns is None:
                self.n_columns = value.shape[1]

        plt.legend()
        plt.grid(True)
        if self.column_labels:
            self.ttl = plt.title(self.column_labels[self.ind])

        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, "Next")
        self.bnext.on_clicked(self._next)
        self.bprev = Button(axprev, "Previous")
        self.bprev.on_clicked(self._prev)

        axcolor = "lightgoldenrodyellow"
        slider_ax = plt.axes([0.1, 0.1, 0.50, 0.03], facecolor=axcolor)
        self.slider = Slider(
            slider_ax, "Column", 0, self.n_columns - 1, valinit=0, valfmt="%.0f"
        )
        self.slider.on_changed(self._update_column)

    def _update_plot(self):
        for line, value in self.lines.items():
            line.set_ydata(value[:, self.ind])
        self.ax.relim()
        # update ax.viewLim using the new dataLim
        self.ax.autoscale()
        # self.ax.autoscale_view()
        if self.column_labels:
            self.ttl.set_text(self.column_labels[self.ind])
        plt.draw()

    def _next(self, event):
        # print("next")
        if self.ind < self.n_columns - 1:
            self.ind += 1
            self.slider.set_val(self.ind)
            self._update_plot()

    def _prev(self, event):
        # print("prev")
        if self.ind > 0:
            self.ind -= 1
            self.slider.set_val(self.ind)
            self._update_plot()

    def _update_column(self, new_column):
        self.ind = int(new_column)
        self._update_plot()
