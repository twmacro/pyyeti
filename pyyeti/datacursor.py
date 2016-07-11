# -*- coding: utf-8 -*-
"""
This small class was inspired by code posted by HappyLeapSecond which
was derived from code posted by Joe Kington. See:

http://stackoverflow.com/questions/13306519/\
get-data-from-plot-with-matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import numpy as np


def form1(x, y, n, ind):
    """Default annotation for plots with 1 line"""
    return 'x: {x:0.2f}\ny: {y:0.2f}'.format(x=x, y=y)


def form2(x, y, n, ind):
    """Default annotation for plots with more than 1 line"""
    return 'x: {x:0.2f}\ny: {y:0.2f}\nline: {n:}'.format(x=x, y=y, n=n)


class DataCursor(object):
    r"""
    Class to show x, y data points and to allow selection of points
    for annotations.

    Attributes
    ----------
    hover : bool
        If True, an annotated large green, semi-transparent dot is
        displayed that follows the mouse as long as the mouse is inside
        the axes.
    form1 : function
        Function to format the x, y data for the annotation. This is
        called for axes that have only one line. Must accept the four
        inputs listed and and return a string. The argument `n` is the
        line number starting at 0 and `ind` is the index into the data
        vectors. `form1` defaults to::

          def form1(x, y, n, ind):
              return 'x: {x:0.2f}\ny: {y:0.2f}'.format(x=x, y=y)

    form2 : function
        Function to format the x, y data for the annotation. This is
        called for axes that have more than one line. Same signature
        as `form1`. Defaults to::

          def form2(x, y, n, ind):
              return ('x: {x:0.2f}\ny: {y:0.2f}\nline: {n:}'.
                     format(x=x, y=y, n=n))

    offsets : tuple
        Two element tuple containing x and y offsets in points for the
        annotation.
    xypoints : list
        Contains [x, y] data pairs for each (non-deleted) left-click.
    inds : list
        Contains index of [x, y] data pairs for each (non-deleted)
        left-click. Same length as `xypoints`.
    lines : list
        Contains line object handles for the selected lines. Same
        length as `xypoints`.
    linenums : list
        Contains line number for each [x, y] data pair, starting at 0.
        Same length as `xypoints`.
    pts : list
        Contains the :func:`plt.scatter` object handles for the selected
        points. Same length as `xypoints`.
    notes : list
        Contains the annotation object handles for the selected points.
        Same length as `xypoints`.

    Notes
    -----
    Having multiple data-cursors active at the same time is
    undesirable. Therefore, this module instantiates one DataCursor
    object called `DC` during the initial import. It is recommended to
    always use `DC` rather than instantiating new DataCursor objects.
    From within a script::

        from datacursor import DC
        ...
        DC.getdata()   # blocks until DataCursor is toggled off (see
                       # table below); or:
        DC.getdata(n)  # blocks until user selects `n` points or the
                       # DataCursor is toggle off

    From an interactive prompt::

        from datacursor import DC
        DC.on()       # doesn't block, but DataCursor is active

    Once the DataCursor is turned on, you'll just mouse over the data
    to see selected data points. These operations are available (when
    the mouse is inside the axes):

    ===========  ======================================================
    Action       Description
    ===========  ======================================================
    left-click   Data point will be stored in the member list
                 `xypoints`. The annotation for points saved will also
                 be saved.
    right-click  Last point is deleted from `xypoints` (along with
                 annotation).
    typing 't'   Toggles operation of DataCursor off and on. Toggling
                 on will reset the `xypoints` data member. Note that
                 toggling on will only work if the DataCursor was on
                 previously for current figure.
    typing 'D'   Deletes last point AND removes the line from the plot
                 via ``line_handle.remove()``. Any older annotations
                 are not deleted.
    ===========  ======================================================

    To get x-y data points from plots from within a script, use
    :func:`DataCursor.getdata`. Enter the number of points or press
    't' to end blocking so the script will continue (see
    :func:`DataCursor.getdata`).

    Once the DataCursor is toggled off, the annotations become
    draggable. Two notes on this:

        1. Due to a bug in matplotlib v1.4.3, dragging annotations
           connected to points with negative x or y coordinates
           doesn't work correctly. Newer versions work properly.
        2. Somehow, annotations sometimes become linked (moving one
           will move another). When that happens, try dragging
           a different annotation; this often breaks the link.

    Interactively, the member functions :func:`DataCursor.on` and
    :func:`DataCursor.off` can be used to turn the DataCursor on and
    off. These functions will update the internal state of the
    `DataCursor` to account for deleted or added items.
    :func:`DataCursor.getdata` calls :func:`DataCursor.on` internally.

    The following example plots some random data, calls
    :func:`DataCursor.getdata` to wait for the user to optionally
    select data points and then toggle the DataCursor off (with
    keystroke 't'). It then prints the selected points.::

        import matplotlib.pyplot as plt
        import numpy as np
        from pyyeti.datacursor import DC
        x = np.arange(500)
        y = np.random.rand(*x.shape)
        fig = plt.figure('demo')
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y)
        DC.getdata()
        print('points are:', DC.xypoints)

    Settings can be changed after instantiation. Here is an example of
    defining a new format for the annotation (for both `form1` and
    `form2`). 3 decimals are used for the x-coordinate and the index
    is included::

        import matplotlib.pyplot as plt
        import numpy as np
        from pyyeti.datacursor import DC

        def formnew(x, y, n, ind):
            return ('x: {x:0.3f}\ny: {y:0.2f}\nindex: {ind:}'
                    .format(x=x, y=y, ind=ind))

        DC.form1 = DC.form2 = formnew

    """

    def __init__(self, ax=None, hover=True,
                 form1=form1, form2=form2,
                 offsets=(-20, 20)):
        """
        Initialize the DataCursor.

        Parameters
        ----------
        ax : axes object(s) or None; optional
            Axes object or list of axes objects as created by
            :func:`matplotlib.pyplot.subplot`. If None, all axes on
            all figures will be automatically included.
        hover : bool; optional
            Sets the `hover` attribute.
        form1 : function; optional
            Sets the `form1` attribute.
        form2 : function; optional
            Sets the `form2` attribute.
        offsets : tuple; optional
            Sets the `offsets` attribute.
        """
        self._ax_input = ax
        self._kid = {}
        self.hover = hover
        self.form1 = form1
        self.form2 = form2
        self.offsets = offsets
        self._is_on = False
        self._in_loop = False
        self._max_points = -1
        self.on()
        self.addptfunc = None
        self.delptfunc = None

    def _init_all(self, errout=False):
        if self._ax_input is None:
            self._figs = [plt.figure(i) for i in plt.get_fignums()]
            if len(self._figs) == 0:
                if errout:
                    raise RuntimeError('no figures; '
                                       'plot something first')
                else:
                    self._ax = []
                    return
            self._ax = [a for fig in self._figs for a in fig.get_axes()]
        else:
            if not isinstance(self._ax_input, list):
                self._ax = [self._ax_input]
            else:
                self._ax = self._ax_input
            self._figs = [a.figure for a in self._ax]
        if len(self._ax) == 0:
            if errout:
                raise RuntimeError('no axes; plot something first')
            else:
                return
        maxlines = 0
        for a in self._ax:
            if len(a.lines) > maxlines:
                maxlines = len(a.lines)
        if maxlines == 0 and errout:
            raise RuntimeError('no lines; plot something first')

    def on(self, newax=-1, callbacks=True):
        """
        Toggles on and (re-)initializes the DataCursor for current
        figures.

        If `newax` is -1, the axes that the DataCursor uses remain
        unchanged (all, if `ax` was originally None). Other values
        reset the DataCursor as described in the main help for the
        `ax` input.
        """
        if newax != -1:
            self._ax_input = newax
        self._init_all()
        self.xypoints = []
        self.inds = []
        self.lines = []
        self.linenums = []
        self.pts = []
        self.notes = []
        self._mid = {}
        self._bid = {}
        self._aid = {}
        self._setup_annotations()
        if callbacks:
            for fig in self._figs:
                cvs = fig.canvas
                if fig not in self._kid:
                    self._kid[fig] = cvs.mpl_connect(
                        'key_press_event', self._key)
                if self.hover:
                    self._mid[fig] = cvs.mpl_connect(
                        'motion_notify_event', self._follow)
                else:
                    self._mid[fig] = None
                self._bid[fig] = cvs.mpl_connect(
                    'button_press_event', self._follow)
                self._aid[fig] = cvs.mpl_connect(
                    'axes_leave_event', self._leave)
            self._is_on = True
        else:
            self._is_on = False

    def off(self):
        """Toggles off the DataCursor and stops it from blocking (if
        :func:`getdata` was called)."""
        self._init_all()
        if self._is_on:
            for ax in self._ax:
                if ax in self._annotation:
                    self._annotation[ax].set_visible(False)
                    self._dot[ax].set_visible(False)
            for fig in self._figs:
                if (self.hover and fig in self._mid and
                        self._mid[fig] is not None):
                    fig.canvas.mpl_disconnect(self._mid[fig])
                if fig in self._bid:
                    fig.canvas.mpl_disconnect(self._bid[fig])
                    fig.canvas.mpl_disconnect(self._aid[fig])
            for an in self.notes:
                # give annotation to figure so it's on top
                ax = an.axes
                ax.figure.texts.append(an)
                ax.texts.pop(ax.texts.index(an))
                # and make it draggable
                offsetbox.DraggableAnnotation(an)
            self._is_on = False
        for fig in self._figs:
            fig.canvas.draw()
        if self._in_loop:
            self._figs[0].canvas.stop_event_loop()
            self._in_loop = False

    def addpt_func(self, func):
        """
        Function to call on a left-click.

        Call signature is: ``func(ax, x, y, n, ind, lineh)``

        The parameters for the function are described below.

        Parameters
        ----------
        ax : axes object
            Axes handle
        x, y : scalars
            The selected ``(x, y)`` data point
        n : integer
            The line number
        ind : integer
            The index of the data point within the plotted data
            vectors
        lineh : line object
            Line handle
        """
        self.addptfunc = func

    def delpt_func(self, func):
        """
        Function to call on a right-click.

        Call signature is: ``func()``
        """
        self.delptfunc = func

    def _add_point(self, ax, x, y, n, ind, lineh):
        self.xypoints.append([x, y])
        self.inds.append(ind)
        self.lines.append(lineh)
        self.linenums.append(n)
        self.pts.append(ax.scatter(x, y, s=130,
                        color='red', alpha=0.4))
        self.notes.append(self._annotation[ax])
        # offsetbox.DraggableAnnotation(self._annotation)
        # make a new annotation box so current one is static
        self._annotation[ax] = self._new_annotation(ax, (x, y))
        if self._in_loop and len(self.xypoints) == self._max_points:
            self.off()
        if self.addptfunc:
            self.addptfunc(ax, x, y, n, ind, lineh)

    def _del_point(self, delete_line=False):
        """Deletes last saved point, if any."""
        if len(self.xypoints) == 0:
            return
        self.xypoints.pop()
        self.inds.pop()
        h = self.lines.pop()
        if delete_line:
            # line may have been deleted already, so catch exception:
            try:
                h.remove()
            except ValueError:
                pass
        self.linenums.pop()
        pt = self.pts.pop()
        pt.remove()
        na = self.notes.pop()
        na.remove()
        if self.delptfunc:
            self.delptfunc()

    def _get_ax(self, event):
        """
        Return axes for event, and possibly modifies event for xdata,
        ydata location.
        """
        if event.inaxes is None:
            return
        if event.inaxes in self._ax:
            ax = event.inaxes
        else:
            # for overlapping axes, only top axes gets the event, so
            # check others manually
            for ax in self._ax:
                pt = (event.x, event.y)
                if ax.contains_point(pt):
                    inv = ax.transData.inverted()
                    event.xdata, event.ydata = inv.transform_point(pt)
                    break
            else:
                return
        return ax

    def _key(self, event):
        """
        Processes 't' key press to toggle the DataCursor on and off.
        """
        if not self._get_ax(event):
            return
        if event.key == 't' or event.key == 'T':
            if self._is_on:
                self.off()
            else:
                self.on()
        elif event.key == 'D':
            self._del_point(delete_line=True)
            event.canvas.draw()

    def _leave(self, event):
        """Event handler for when mouse leaves axes."""
        ax = self._get_ax(event)
        if not ax:
            return
        self._annotation[ax].set_visible(False)
        self._dot[ax].set_visible(False)
        event.canvas.draw()

    def _follow(self, event):
        """Event handler for when mouse is moved insided axes and for
        left and right clicks."""
        ax = self._get_ax(event)
        if not ax:
            return
        x, y = event.xdata, event.ydata
        x, y, n, ind, lineh = self._snap(ax, x, y)
        if x is None:
            return
        annotation = self._annotation[ax]
        dot = self._dot[ax]
        annotation.xy = x, y
        if len(ax.lines) > 1:
            annotation.set_text(self.form2(x, y, n, ind))
        else:
            annotation.set_text(self.form1(x, y, n, ind))
        annotation.set_visible(True)
        dot.set_offsets((x, y))
        dot.set_visible(True)
        if event.name == 'button_press_event':
            if event.button == 1:
                self._add_point(ax, x, y, n, ind, lineh)
            elif event.button == 3 and len(self.xypoints) > 0:
                self._del_point()
        event.canvas.draw()

    def _new_annotation(self, ax, xy):
        bbox = dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.75)
        arrow = dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        return ax.annotate('', xy=xy,
                           ha='right',
                           va='bottom',
                           xytext=self.offsets,
                           textcoords='offset points',
                           bbox=bbox, arrowprops=arrow,
                           visible=False)

    def _setup_annotations(self):
        """Create the annotation boxes. The string value and the
        position will be set for each event."""
        self._annotation = {}
        self._dot = {}
        for ax in self._ax:
            if len(ax.lines) > 0:
                ln = ax.lines[0]
                xy = ln.get_xdata()[0], ln.get_ydata()[0]
            else:
                xy = 0, 0
            self._annotation[ax] = self._new_annotation(ax, xy)
            self._dot[ax] = ax.scatter(xy[0], xy[1], s=130,
                                       color='green',
                                       alpha=0.7,
                                       visible=False)

    def _snap(self, ax, x, y):
        """Return the value in self._points closest to x, y."""
        dmin = np.inf
        xyn = None, None, None
        for n, ln in enumerate(ax.lines):
            xdata = ln.get_xdata()
            ydata = ln.get_ydata()
            dx = (xdata-x) / np.diff(ax.get_xlim())
            dy = (ydata-y) / np.diff(ax.get_ylim())
            # dx = (xdata-x)
            # dy = (ydata-y)
            d = np.sqrt(dx**2. + dy**2.)
            ind = np.argmin(d)
            if d[ind] < dmin:
                dmin = d[ind]
                xyn = xdata[ind], ydata[ind], n, ind, ln
        return xyn

    def _fake_getdata(self):
        """
        Available to bypass actual mouse interaction while running
        tests of routines that use :func:`getdata`. If this function
        returns anything other than None, :func:`getdata` is bypassed.
        """
        return

    def getdata(self, maxpoints=-1,
                msg='Select points, hit "t" inside axes when done'):
        """
        Suspend python while user selects points up to `maxpoints`.
        If `maxpoints` is < 0, the loop will last until user hits 't'
        inside the axes ('t' toggles off the DataCursor).
        """
        # _fake_getdata is used for testing functions that need
        # getdata
        if self._fake_getdata() is None:
            if maxpoints == 0:
                return
            if maxpoints < 0 and msg:
                print(msg)
            self.on()
            self._in_loop = True
            self._max_points = maxpoints
            self._figs[0].canvas.start_event_loop(timeout=-1)


# instantiate one object, meant for general use:
DC = DataCursor()
