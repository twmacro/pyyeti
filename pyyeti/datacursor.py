# -*- coding: utf-8 -*-
"""
A class for interactively annotating matplotlib plots.

This class was inspired by code posted by HappyLeapSecond which was
derived from code posted by Joe Kington. See:

http://stackoverflow.com/questions/13306519/\
get-data-from-plot-with-matplotlib
"""

from types import SimpleNamespace
import numbers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d


from time import sleep


def mk_label(point):
    """Default annotation function"""

    def tostr(xyz):
        if isinstance(xyz, numbers.Number):
            return f"{xyz:.5g}"
        return f"{xyz}"

    if point.ax.name == "polar":
        deg = np.rad2deg(point.x % (2 * np.pi))
        label = f"θ: {tostr(point.x)} ({tostr(deg)}°)\nrad: {tostr(point.y)}"
    else:
        label = f"x: {tostr(point.x)}\ny: {tostr(point.y)}"
        if point.z is not None:
            label += f"\nz: {tostr(point.z)}"

    if point.nlines > 1:
        label += f"\n{point.handle.get_label()}"

    return label


def _ensure_iterable(a):
    if not isinstance(a, (list, tuple)):
        return [a]
    return a


class DataCursor(object):
    r"""
    Class to show x, y data points and to allow selection of points
    for annotations.

    Attributes
    ----------
    hover : bool
        If True, an annotated large green, semi-transparent dot is
        displayed that follows the mouse as long as the mouse is
        inside the axes. Note: setting this directly is possible; you
        just have to turn the DataCursor off and back on for the
        setting to take effect. For example::

            from pyyeti.datacursor import DC
            DC.hover = False
            DC.off()
            DC.on()

    mk_label : function
        Function that returns an annotation label for a data
        point. Must accept a single SimpleNamespace argument that
        contains information about the selected data point. `mk_label`
        defaults to::

            def mk_label(point):

                def tostr(xyz):
                    if isinstance(xyz, numbers.Number):
                        return f"{xyz:.5g}"
                    return f"{xyz}"

                if point.ax.name == "polar":
                    deg = np.rad2deg(point.x % (2 * np.pi))
                    label = (
                        f"θ: {tostr(point.x)} ({tostr(deg)}°)\n"
                        f"rad: {tostr(point.y)}"
                    )
                else:
                    label = f"x: {tostr(point.x)}\ny: {tostr(point.y)}"
                    if point.z is not None:
                        label += f"\nz: {tostr(point.z)}"

                if point.nlines > 1:
                    label += f"\n{point.handle.get_label()}"

                return label

        The SimpleNamespace ``point`` contains the following
        attributes:

        ======= ======================================================
         attr   Description
        ======= ======================================================
        ax      Axes handle
        handle  Line2D or PathCollection handle for line
        index   Index into the data vectors for data point
        lines   Total number of lines on axes
        n       Line number starting at 0
        x,y,z   x, y, z coordinates of data point; z is None for 2D
        dot     The :func:`matplotlib.pyplot.scatter` (PathCollection)
                object handle. Note that :class:`DataCursor` ignores
                these added annotation points when moused over; they
                are identified by the added attribute
                "_pyyeti_dc_point".
        note    The annotation object handle from
                :func:`matplotlib.pyplot.annotate`.
        xy_note The (x, y) coordinates for the note.
        ======= ======================================================

    offsets : tuple
        Two element tuple containing x and y offsets in points for the
        annotation.
    bbox : dict; optional
        Defines the `bbox` parameter for
        :meth:`matplotlib.axes.Axes.annotate`
    arrowprops : dict; optional
        Defines the `arrowprops` parameter for
        :meth:`matplotlib.axes.Axes.annotate`
    followdot : dict; optional
        Typically defines the `s`, `color`, and `alpha` settings (and
        possibly others as desired) for
        :meth:`matplotlib.axes.Axes.scatter`. That function is used
        for drawing the "dot" on the plot that follows the mouse and
        highlights the currently selected data point.
    permdot : dict; optional
        Similar to `followdot` except this is a "permanent" dot;
        this gets placed after left clicking.
    points : list
        Contains list of SimpleNamespace objects. Each "point" is as
        described above under the `mk_label` attribute.
    xyz : 2d ndarray or None
        If points have been selected and the datacursor has been
        turned off, the `xyz` attribute will be filled with the x, y,
        z data for the selected points for all axes. If there are no
        3d plots, only the x and y columns will be present. The order
        of the rows follows the selection order (same order as
        `points`; the source of this data).

    Notes
    -----
    Having multiple data-cursors active at the same time is
    undesirable. Therefore, this module instantiates one DataCursor
    object called `DC` during the initial import. It is recommended to
    always use `DC` rather than instantiating new DataCursor objects.
    From within a script::

        from datacursor import DC
        ...
        DC.getdata()   # blocks until DataCursor is turned off via 't'
        DC.getdata(n)  # blocks until user selects `n` points (or
                       # DataCursor is turned off via 't')

    From an interactive prompt::

        from datacursor import DC
        DC.on()       # doesn't block, but DC is active

    Once the DataCursor is turned on, you'll just mouse over the data
    to see selected data points. These operations are available (when
    the mouse is inside the axes):

    ===========  ======================================================
    Action       Description
    ===========  ======================================================
    left-click   Data point will be stored in the member list
                 `points`.
    right-click  Last point is deleted from the plot and from `points`.
    typing 't'   Turns off DataCursor. To turn on, use ``DC.on``. Note
                 that turning on will reset `points`.
    typing 'D'   Deletes last point AND removes the line from the plot
                 via ``line_handle.remove()``. Any older annotations
                 are not deleted.
    ===========  ======================================================

    To get data points from plots from within a script, use
    :func:`DataCursor.getdata`. Enter the number of points or press
    't' to end blocking so the script will continue (see
    :func:`DataCursor.getdata`).

    Once the DataCursor is turned off, the annotations become
    draggable. Note that, at least for some versions of Matplotlib,
    annotations sometimes become linked (moving one will move
    another). When that happens, try dragging a different annotation;
    this sometimes breaks the link.

    Interactively, the member functions :func:`DataCursor.on` and
    :func:`DataCursor.off` are used to turn the DataCursor on and
    off. These functions will update the internal state of the
    `DataCursor` to account for deleted or added items.
    :func:`DataCursor.getdata` calls :func:`DataCursor.on` internally.

    The following example plots some random data, calls
    :func:`DataCursor.getdata` to wait for the user to optionally
    select data points and then turn the DataCursor off (with
    keystroke 't'). It then prints the selected points::

        import matplotlib.pyplot as plt
        import numpy as np
        from pyyeti.datacursor import DC
        rng = np.random.default_rng()
        x = np.arange(500)/250
        y = rng.uniform(size=x.shape)
        fig = plt.figure('demo')
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y)
        DC.getdata()
        # use DC.pause if you want to drag boxes around before
        # continuing:
        DC.pause()
        print('x, y values of selected points are:')
        print(np.array([[p.x, p.y] for p in DC.points]))

    Settings can be changed after instantiation. Here is an example of
    defining a new format for the annotation. Only the line label is
    included in the annotation. The example also changes the permanent
    dot to a gray pentagon::

        import matplotlib.pyplot as plt
        import numpy as np
        from pyyeti.datacursor import DC

        def new_label(point):
            return (f'{point.handle.get_label()}\n'
                    f'({point.x},{point.y:.2f})')

        DC.mk_label = new_label
        DC.permdot = dict(s=130, color='black', alpha=0.4,
                          marker='p')

        rng = np.random.default_rng()
        plt.plot(rng.normal(size=50), label='Gaussian')
        plt.plot(rng.uniform(size=50), label='Uniform')
        DC.on()

    For increased versatility, there are two optional functions the
    user can define that will be called when a point is added
    (left-click) and when a point is deleted (right-click). See
    :func:`DataCursor.addpt_func` and :func:`DataCursor.delpt_func`
    for more information on the call signatures. Here is a simple
    example that just prints statements to the screen::

        import matplotlib.pyplot as plt
        import numpy as np
        from pyyeti.datacursor import DC

        def addpt(point):
            print(f'You selected ({point.x}, {point.y}, {point.z})')

        def delpt(point):
            print(f'You deleted ({point.x}, {point.y}, {point.z})')

        DC.addpt_func(addpt)
        DC.delpt_func(delpt)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        rng = np.random.default_rng()
        coords = rng.normal(size=(6, 3))
        dots = ax.scatter(*coords.T)
        ax.plot(*(coords.T + 0.1), "v")
        DC.on()
    """

    def __init__(
        self,
        ax=None,
        figs=None,
        hover=True,
        mk_label=mk_label,
        offsets=(-20, 20),
        bbox=dict(boxstyle="round,pad=0.5", fc="gray", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        followdot=dict(s=130, color="green", alpha=0.7),
        permdot=dict(s=130, color="red", alpha=0.4),
    ):
        """
        Initialize the DataCursor.

        Parameters
        ----------
        ax : axes object(s) or None; optional
            Axes object or list of axes objects as created by
            :func:`matplotlib.pyplot.subplot` (for example). If None,
            all axes on all selected figures will be automatically
            included. Takes precedence over the `figs` input.
        figs : figure object(s) or None; optional
            Alternative to the `ax` input. If `ax` is not input,
            `figs` specifies a figure object or list of figure objects
            as created by :func:`matplotlib.pyplot.figure`. If None,
            all applicable figures will be automatically included.
        hover : bool; optional
            Sets the `hover` attribute.
        form1 : function; optional
            Sets the `form1` attribute.
        form2 : function; optional
            Sets the `form2` attribute.
        offsets : tuple; optional
            Sets the `offsets` attribute.
        bbox : dict; optional
            Sets the `bbox` attribute.
        arrowprops : dict; optional
            Sets the `arrowprops` attribute.
        followdot : dict; optional
            Sets the `followdot` attribute.
        permdot : dict; optional
            Sets the `permdot` attribute.
        """
        self._ax_input = ax
        self._figs_input = figs
        self.hover = hover
        self.mk_label = mk_label
        self.offsets = offsets
        self.bbox = bbox
        self.arrowprops = arrowprops
        self.followdot = followdot
        self.permdot = permdot
        self._is_on = False
        self._in_loop = False
        self._max_points = -1
        self.on()
        self.addptfunc = None
        self.delptfunc = None

    def _init_all(self, errout=False):
        if self._ax_input is None:
            if self._figs_input is None:
                self._figs = [plt.figure(i) for i in plt.get_fignums()]
            else:
                self._figs = _ensure_iterable(self._figs_input)
            if len(self._figs) == 0:
                if errout:
                    raise RuntimeError("no figures; plot something first")
                else:
                    self._ax = []
                    return
            self._ax = [a for fig in self._figs for a in fig.get_axes()]
        else:
            self._ax = _ensure_iterable(self._ax_input)
            self._figs = [a.figure for a in self._ax]
        if len(self._ax) == 0:
            if errout:
                raise RuntimeError("no axes; plot something first")
            else:
                return
        maxlines = 0
        for a in self._ax:
            handles = self._get_data_handles(a)
            if len(handles) > maxlines:
                maxlines = len(handles)
        if maxlines == 0 and errout:
            raise RuntimeError("no lines; plot something first")

    def on(self, ax=None, figs=None, callbacks=True, reset=True):
        """
        Turns on and (re-)initializes the DataCursor for current
        figures.

        Parameters
        ----------
        ax : axes object(s) or None or -1; optional
            Axes object or list of axes objects as created by
            :func:`matplotlib.pyplot.subplot` (for example). If None,
            all axes on all selected figures will be automatically
            included. Takes precedence over the `figs` input. If -1,
            leave this setting as specified during instantiation.
        figs : figure object(s) or None or -1; optional
            Alternative to the `ax` input. If `ax` is not input,
            `figs` specifies a figure object or list of figure objects
            as created by :func:`matplotlib.pyplot.figure`. If None,
            all applicable figures will be automatically included. If
            -1, leave this setting as specified during instantiation.
        callbacks : bool or str; optional
            If False, call-backs are not turned on. If True, all call-
            backs are turned on. If 'key_only', only the key-press
            call-back is turned on (used for pausing; see
            :func:`pause`).
        reset : bool; optional
            If True, the `points` and other data members are reset
            to empty lists. Otherwise, if `reset` is False, your new
            data will be appended on to your previous data.

        Returns
        -------
        None
        """
        if ax != -1:
            self._ax_input = ax
        if figs != -1:
            self._figs_input = figs
        self._init_all()
        if reset:
            self.points = []
            self.xyz = None
        self._kid = {}  # key press event
        self._mid = {}  # motion event
        self._bid = {}  # button press event
        self._aid = {}  # axes leave event
        self._fig_layout_engine = {}
        self._setup_annotations()
        if callbacks:
            for fig in self._figs:
                cvs = fig.canvas
                self._kid[fig] = cvs.mpl_connect("key_press_event", self._key)
                if callbacks != "key_only":
                    if self.hover:
                        self._mid[fig] = cvs.mpl_connect(
                            "motion_notify_event", self._follow
                        )
                        if hasattr(fig, "get_layout_engine"):
                            self._fig_layout_engine[fig] = fig.get_layout_engine()
                            fig.set_layout_engine("none")
                    else:
                        self._mid[fig] = None
                    self._bid[fig] = cvs.mpl_connect("button_press_event", self._follow)
                    self._aid[fig] = cvs.mpl_connect("axes_leave_event", self._leave)
            self._is_on = True
        else:
            self._is_on = False

    def off(self, stop_blocking=True):
        """
        Turns off the DataCursor and optionally stops it from blocking

        Parameters
        ----------
        stop_blocking : bool; optional
            If True, have the data cursor stop blocking so Python can
            continue with whats next. Otherwise, if `stop_blocking` is
            False, Python will wait; this is probably only useful when
            the data cursor is controlled in a GUI environment.

        Notes
        -----
        Note that the keystroke 't' will also turn off the DataCursor;
        in that case, `stop_blocking` is True.
        """
        self._ax_input = None
        self._figs_input = None
        self._init_all()
        if self._is_on:
            for ax in self._ax:
                if ax in self._annotation:
                    self._annotation[ax].set_visible(False)
                    self._mk_followdot_invisible(ax)
            for fig in self._figs:
                if fig in self._kid:
                    fig.canvas.mpl_disconnect(self._kid[fig])
                if self.hover:
                    if fig in self._mid and self._mid[fig] is not None:
                        fig.canvas.mpl_disconnect(self._mid[fig])
                    if fig in self._fig_layout_engine:
                        fig.set_layout_engine(self._fig_layout_engine[fig])
                if fig in self._bid:
                    fig.canvas.mpl_disconnect(self._bid[fig])
                    fig.canvas.mpl_disconnect(self._aid[fig])
            # make annotations draggable:
            for pt in self.points:
                pt.note.draggable()
            self._is_on = False
        for fig in self._figs:
            fig.canvas.draw()
        if self._in_loop and stop_blocking:
            self._figs[0].canvas.stop_event_loop()
            self._in_loop = False

        # make xyz:
        if any(p.z for p in self.points):
            xyz = [[p.x, p.y, p.z if p.z is not None else np.nan] for p in self.points]
        else:
            xyz = [[p.x, p.y] for p in self.points]
        self.xyz = np.array(xyz)

    def addpt_func(self, func):
        """
        Function to call on a left-click.

        Call signature is: ``func(point)``

        The input ``point`` is a SimpleNamespace as described for the
        `mk_label` attribute. See :class:`DataCursor`.
        """
        self.addptfunc = func

    def delpt_func(self, func):
        """
        Function to call on a right-click.

        Call signature is: ``func(point)``

        The input ``point`` is a SimpleNamespace as described for the
        `mk_label` attribute. See :class:`DataCursor`.
        """
        self.delptfunc = func

    @staticmethod
    def _add_annotation_point(ax, x, y, dct, vis):
        # get/set axis limits; this is strange, but ax.scatter will
        # rescale the plots for some reason in some cases
        for lim in ("xlim", "ylim", "zlim"):
            if fnc := getattr(ax, f"get_{lim}", None):
                getattr(ax, f"set_{lim}")(fnc())
        h = ax.scatter(x, y, **dct, visible=vis)
        h._pyyeti_dc_point = True
        return h

    def _mk_followdot_invisible(self, ax):
        if self._dot[ax] is not None:
            self._dot[ax].set_visible(False)

    def _get_followdot_handle(self, ax, x, y):
        if self._dot[ax] is None:
            self._dot[ax] = self._add_annotation_point(ax, x, y, self.followdot, False)
        else:
            self._dot[ax].set_offsets((x, y))
        return self._dot[ax]

    @staticmethod
    def _is3d(ax):
        return hasattr(ax, "get_proj")

    def _add_point(self, point):
        # if 3d, set visible to False (i don't know how to get proper
        # coordinates for the dot):
        vis = False if self._is3d(point.ax) else True
        x, y = point.xy_note
        point.dot = self._add_annotation_point(point.ax, x, y, self.permdot, vis)
        point.note = self._annotation[point.ax]
        self.points.append(point)

        # make a new annotation box so current one is static
        self._annotation[point.ax] = self._new_annotation(point.ax, (x, y))
        if self._in_loop and len(self.points) == self._max_points:
            self.off()
        if self.addptfunc:
            self.addptfunc(point)

    def _del_point(self, ax=None, delete_line=False):
        """Deletes last saved point, if any."""
        if len(self.points) == 0:
            return

        if ax:
            for i in reversed(range(len(self.points))):
                if self.points[i].ax == ax:
                    point = self.points.pop(i)
                    break
            else:
                return

        if delete_line:
            # line may have been deleted already, so catch exception:
            line = point.handle
            try:
                line.remove()
            except ValueError:
                pass

        point.dot.remove()
        point.note.remove()
        if self.delptfunc:
            self.delptfunc(point)

    def _get_ax(self, event):
        """
        Return axes for event, and possibly modifies event for xdata,
        ydata location.
        """
        if event.inaxes is None:
            return
        if event.inaxes in self._ax:
            return event.inaxes

    def _key(self, event):
        """
        Processes 't' key press to turn the DataCursor off.
        """
        if not self._get_ax(event):
            return
        if event.key == "t" or event.key == "T":
            if self._is_on:
                self.off()
            else:
                self.on()
        elif event.key == "D":
            self._del_point(delete_line=True)
            event.canvas.draw()

    def _leave(self, event):
        """Event handler for when mouse leaves axes."""
        ax = self._get_ax(event)
        if not ax:
            return
        self._annotation[ax].set_visible(False)
        self._mk_followdot_invisible(ax)
        event.canvas.draw()

    def _follow(self, event):
        """Event handler for when mouse is moved insided axes and for
        left and right clicks."""
        ax = self._get_ax(event)
        if not ax:
            return
        point = self._snap(ax, event.x, event.y)
        if point is None:
            return
        x, y = point.xy_note
        annotation = self._annotation[ax]
        annotation.xy = x, y
        annotation.set_text(self.mk_label(point))
        dot = self._get_followdot_handle(ax, x, y)
        if event.name == "button_press_event":
            if event.button == 1:
                annotation.set_visible(True)
                self._add_point(point)
            elif event.button == 3 and len(self.points) > 0:
                self._del_point(ax=ax)
        elif self.hover:
            if not self._is3d(ax):
                dot.set_visible(True)
            annotation.set_visible(True)
        event.canvas.draw()

    def _new_annotation(self, ax, xy):
        return ax.annotate(
            "",
            xy=xy,
            ha="right",
            va="bottom",
            xytext=self.offsets,
            textcoords="offset points",
            bbox=self.bbox,
            arrowprops=self.arrowprops,
            visible=False,
        )

    @staticmethod
    def _get_data_handles(ax):
        return [
            child
            for child in ax.get_children()
            if isinstance(child, (PathCollection, Line2D))
            and not hasattr(child, "_pyyeti_dc_point")
        ]

    @staticmethod
    def _get_xy_data(ax, h):
        if isinstance(h, Line2D):
            x = h.get_xdata()
            y = h.get_ydata()
        elif hasattr(h, "_offsets3d"):
            x, y, _ = proj3d.proj_transform(*np.array(h._offsets3d), ax.get_proj())
        else:
            x, y = (*h.get_offsets().data.T,)
        return x, y

    def _get_xy_data_display(self, ax, h):
        if isinstance(h, Line2D):
            x_ann, y_ann = h.get_xdata(), h.get_ydata()
            x = ax.convert_xunits(x_ann)  # eg, convert dates to floats
            y = ax.convert_yunits(y_ann)
        elif hasattr(h, "_offsets3d"):
            x, y, _ = proj3d.proj_transform(*np.array(h._offsets3d), ax.get_proj())
            x_ann, y_ann = x, y
        else:
            x, y = (*h.get_offsets().data.T,)
            x_ann, y_ann = x, y

        # get pixel coordinates:
        x, y = ax.transData.transform(np.column_stack((x, y))).T
        return x, y, x_ann, y_ann

    @staticmethod
    def _get_xyz(h, ind):
        """
        Get original 3D location of a point, instead of the mapped-to-2D
        x, y location
        """
        if isinstance(h, Line2D):
            return np.array(h.get_data_3d())[:, ind]
        return np.array(h._offsets3d)[:, ind]

    def _setup_annotations(self):
        """Create the annotation boxes. The string value and the
        position will be set for each event."""
        self._annotation = {}
        self._dot = {}

        for ax in self._ax:
            xl, yl = ax.get_xlim(), ax.get_ylim()
            xy = sum(xl) / 2, sum(yl) / 2
            self._annotation[ax] = self._new_annotation(ax, xy)
            self._dot[ax] = None

    @staticmethod
    def _scalars(*args):
        return [v.item() if isinstance(v, np.ndarray) else v for v in args]

    def _snap(self, ax, x, y):
        """Return the plotted value closest to x, y."""
        dmin = np.inf
        lines = self._get_data_handles(ax)
        nlines = len(self._get_data_handles(ax))
        if nlines == 0:
            return None
        best = None

        for n, h in enumerate(lines):
            x_pix, y_pix, x_ann, y_ann = self._get_xy_data_display(ax, h)
            dx = x_pix - x
            dy = y_pix - y
            d = dx**2.0 + dy**2.0

            # use try block in case data is all NaNs:
            try:
                ind = np.nanargmin(d)
            except ValueError:
                pass
            else:
                if d[ind] < dmin:
                    try:
                        xy_note = self._scalars(x_ann[ind], y_ann[ind])
                    except ValueError:
                        pass
                    else:
                        dmin = d[ind]
                        best = n, ind, h

        if best is None:
            return None
        n, ind, handle = best
        # get original data coordinates
        if self._is3d(ax):
            xo, yo, zo = self._get_xyz(handle, ind)
        else:
            xo, yo = handle.get_xydata()[ind]
            zo = None
        point = SimpleNamespace(
            x=xo,
            y=yo,
            z=zo,
            ax=ax,
            n=n,
            index=ind,
            handle=handle,
            nlines=nlines,
            xy_note=xy_note,
        )
        return point

    def _fake_getdata(self):
        """
        Available to bypass actual mouse interaction while running
        tests of routines that use :func:`getdata`. If this function
        returns anything other than None, :func:`getdata` is bypassed.
        """
        return

    def getdata(self, maxpoints=-1, msg='Select points, hit "t" inside axes when done'):
        """
        Suspend python while user selects points up to `maxpoints`.
        If `maxpoints` is < 0, the loop will last until user hits 't'
        inside the axes ('t' turns off the DataCursor). Deleted points
        are not counted.
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

    def pause(self, msg='Pausing, hit "t" inside axes to continue'):
        """
        Suspend python so user can interact with plots (such as moving
        previously added annotations) before continuing. Hit 't'
        inside the axes to continue.
        """
        print(msg)
        self.on(callbacks="key_only", reset=False)
        self._in_loop = True
        self._figs[0].canvas.start_event_loop(timeout=-1)


# instantiate one object, meant for general use:
DC = DataCursor()
