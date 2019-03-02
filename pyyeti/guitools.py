"""
GUI tools using tkinter. The multicolumn list is inspired by a post on
Stackoverflow:

http://stackoverflow.com/questions/5286093/\
display-listbox-with-columns-using-tkinter
"""
import os
import tkinter as tk
import tkinter.filedialog as filedialog
import tkinter.font as tkFont
import tkinter.ttk as ttk

LASTOPENDIR = None
LASTSAVEDIR = None

__all__ = [
    "get_file_name",
    "askopenfilename",
    "asksaveasfilename",
    "MultiColumnListbox",
]


def get_file_name(f, read):
    """
    Utility function to get the file name using guitools if needed.

    Parameters
    ----------
    f : string or None
        File name or directory name or None. If `f` is not None or is
        not a directory name, this routine just returns it as is. If
        it is a directory name, it is used as the `initialdir` option
        to :func:`askopenfilename` or :func:`asksaveasfilename`.
    read : bool
        If True, calls :func:`askopenfilename` to get file name.
        Otherwise, :func:`asksaveasfilename` is called.

    Returns
    -------
    filename : string
        The selected filename.
    """
    if isinstance(f, str) and os.path.isdir(f):  # pragma: no cover
        initialdir = f
        f = None
    else:
        initialdir = None
    if f is None:  # pragma: no cover
        if read:
            return askopenfilename(initialdir=initialdir)
        else:
            return asksaveasfilename(initialdir=initialdir)
    return f


def askopenfilename(title=None, filetypes=None, initialdir=None):  # pragma: no cover
    """
    Use GUI to select file for reading

    Parameters
    ----------
    title : string or None
        Title of window. Use None to accept tkinter default.
    filetypes : list or None
        Option to limit file search to certain patterns. Typically,
        this is a list of tuples, each tuple containing a description
        followed by a pattern (see example below).
    initialdir : string or None
        Initial directory to begin search. If None, use previous
        location if there is one.

    Returns
    -------
    filename : string
        The selected filename.

    Notes
    -----
    Here is a simple example::

        from pyyeti import guitools
        filename = guitools.askopenfilename()

    To filter the files to selected types::

        from pyyeti import guitools
        filetypes = [('Output4 files', '*.op4'),
                     ('All files', '*')]
        filename = guitools.askopenfilename(filetypes=filetypes)

    """
    global LASTOPENDIR
    root = tk.Tk()
    root.withdraw()
    if filetypes is None:
        filetypes = []
    if initialdir is None:
        initialdir = LASTOPENDIR
    filename = filedialog.askopenfilename(
        parent=root, filetypes=filetypes, initialdir=initialdir, title=title
    )
    root.destroy()
    if filename:
        LASTOPENDIR = os.path.dirname(filename)
    return filename


def asksaveasfilename(title=None, filetypes=None, initialdir=None):  # pragma: no cover
    """
    Use GUI to select file for writing

    Parameters
    ----------
    title : string or None
        Title of window. Use None to accept tkinter default.
    filetypes : list or None
        Option to limit file search to certain patterns. Typically,
        this is a list of tuples, each tuple containing a description
        followed by a pattern (see example below).
    initialdir : string or None
        Initial directory to begin search. If None, use previous
        location if there is one.

    Returns
    -------
    filename : string
        The selected filename.

    Notes
    -----
    Here is a simple example::

        from pyyeti import guitools
        filename = guitools.asksaveasfilename()

    To filter the files to selected types::

        from pyyeti import guitools
        filetypes = [('Output4 files', '*.op4'),
                     ('All files', '*')]
        filename = guitools.asksaveasfilename(filetypes=filetypes)

    """
    global LASTSAVEDIR
    root = tk.Tk()
    root.withdraw()
    if filetypes is None:
        filetypes = []
    if initialdir is None:
        initialdir = LASTSAVEDIR
    filename = filedialog.asksaveasfilename(
        parent=root, filetypes=filetypes, initialdir=initialdir, title=title
    )
    # self.root.after_idle(self._quit)
    root.destroy()
    if filename:
        LASTSAVEDIR = os.path.dirname(filename)
    return filename


class MultiColumnListbox(object):  # pragma: no cover
    """
    Use a ttk.TreeView to build a linked, multicolumn listbox.

    Once a window is created, you may select a single item by either
    selecting a row and pressing "Done" or by double-clicking a row.
    Or, you may select multiple items and pressing "Done".

    You can filter the values shown by entering strings in the filter
    boxes above the multicolumn listbox and hitting the Return
    key. All values in a row must match their respective filter for
    the row to remain visible. A value matches if it contains the
    filter anywhere in it.

    After selection, retrieve your selection by accessing attributes
    `sel_index` or `sel_dict`. `sel_index` is a list of indexes into
    the provided lists. `sel_dict` is a dictionary with the keys being
    the indexes and the values being another dictionary of `header` :
    list-item pairs.

    Example code::

        from pyyeti import guitools
        headers = ['First', 'Middle', 'Last']
        lst1 = ['Tony', 'Jennifer', 'Albert', 'Marion']
        lst2 = ['J.', 'M.', 'E.', 'K.']
        lst3 = ['Anderson', 'Smith', 'Kingsley', 'Cotter']
        ind = guitools.MultiColumnListbox(
                 'Select person', headers, [lst1, lst2, lst3]
              ).sel_index[0]
        print(f'First Person is {lst1[ind]} {lst2[ind]} {lst3[ind]}')

    Or, using the `sel_dict` attribute::

        dct = guitools.MultiColumnListbox(
                 'Select person', headers, [lst1, lst2, lst3]
              ).sel_dict
        key = sorted(dct)[0]
        vals = dct[key]
        print(f"First Person is {vals['First']} {vals['Middle']} "
              f"{vals['Last']}")

    """

    def __init__(
        self,
        title,
        headers,
        lists,
        topstring=(
            "Click on header to sort by that column;\n"
            "Drag boundary to change width of column"
        ),
    ):
        """
        Initialize a :class:`MultiColumnListbox` instance.

        Parameters
        ----------
        title : string
            Title for the window
        headers : list of strings
            List of column headers
        lists : list of lists
            Corresponds to `headers`. Each list must be the same
            length and contain the contents of the columns. The
            contents are expected to be strings.
        topstring : string; optional
            String to print above table
        """
        self.root = tk.Tk()
        self.root.title(title)
        self.tree = None
        self.headers = headers
        self.lists = lists
        self.topstring = topstring
        self.sel_dict = {}
        self.sel_index = []
        self.detached_items = [0] * len(lists[0])
        self._setup_widgets()
        self._build_tree()
        self.root.mainloop()

    def _setup_widgets(self):
        msg = tk.Text(wrap="word", height=2, font="TkDefaultFont")
        msg.insert("1.0", self.topstring)  # line 1, column 0
        msg.configure(bg=self.root.cget("bg"), relief="flat", state="disabled")
        msg.pack(fill="x")
        self._add_filter_boxes()
        container = ttk.Frame()
        container.pack(fill="both", expand=True)
        # create a treeview with dual scrollbars
        self.tree = ttk.Treeview(
            height=min(25, len(self.lists[0])), columns=self.headers, show="headings"
        )
        vsb = ttk.Scrollbar(orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(column=0, row=0, sticky="nsew", in_=container)
        vsb.grid(column=1, row=0, sticky="ns", in_=container)
        hsb.grid(column=0, row=1, sticky="ew", in_=container)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)
        # add a done button:
        button = ttk.Button(text="Done", command=self._get_selection)
        button.pack()

    def _add_filter_boxes(self):
        container = ttk.Frame()
        container.pack(fill="x")
        msg = ttk.Label(
            container,
            text="Filters:",
            # background='gray',
            borderwidth=5,
        )
        # msg.grid(row=0, rowspan=2, column=0, padx=10)
        msg.grid(row=0, column=0, padx=10)

        self.casesen = tk.IntVar()
        self.casesen.set(0)
        cbutton = ttk.Checkbutton(
            container, text="Case Sensitive", variable=self.casesen
        )  # command=self._checkbutton_action)
        cbutton.grid(row=1, column=0)

        self.filter_var = []
        # add an Entry widget for all columns except "Index":
        for i, header in enumerate(self.headers):
            filter_var = tk.StringVar()
            filter_entry = ttk.Entry(container, textvariable=filter_var)
            filter_var.set("")
            filter_entry.bind("<Key-Return>", self._apply_filters)
            # filter_entry.pack(side='left', padx=10, expand=True)
            ttk.Label(container, text=header).grid(row=0, column=i + 1, sticky="w")
            filter_entry.grid(row=1, column=i + 1, sticky="w")
            # container.grid_columnconfigure(i+1, weight=1)
            self.filter_var.append(filter_var)

    def _apply_filters(self, event=None):
        # detach all items remaining:
        for item in self.tree.get_children(""):
            i = int(item[1:], 16) - 1
            self.detached_items[i] = item
            self.tree.detach(item)
        # reattach those where all filters pass:
        if self.casesen.get():
            filtervars = [v.get() for v in self.filter_var]

            def _do_find(value, sub):
                return value.find(sub) > -1

        else:
            filtervars = [v.get().lower() for v in self.filter_var]

            def _do_find(value, sub):
                return value.lower().find(sub) > -1

        for i in range(len(self.lists[0])):
            for string, curlist in zip(filtervars, self.lists):
                if string and not _do_find(curlist[i], string):
                    break
            else:
                # only here if the 'break' was not executed ...
                # when all filters pass:
                self.tree.move(self.detached_items[i], "", i)
                self.detached_items[i] = 0

    def _quit(self):
        # self.root.quit()
        self.root.destroy()

    def _store_selection(self, items):
        dct = {}
        index = []
        for item in items:
            i = int(item[1:], 16) - 1
            dct[i] = self.tree.set(item)
            index.append(i)
        self.sel_dict = dct
        self.sel_index = index
        self.root.after_idle(self._quit)

    def _get_selection(self):
        self._store_selection(self.tree.selection())

    def _double_click(self, event):
        item = self.tree.identify("item", event.x, event.y)
        self._store_selection((item,))

    def _build_tree(self):
        for col in self.headers:
            self.tree.heading(
                col, text=col.title(), command=lambda c=col: _sortby(self.tree, c, 0)
            )
            # adjust the column's width to the header string
            # - add 15 pixels for a little buffer
            self.tree.column(col, width=tkFont.Font().measure(col.title()) + 15)

        for item in zip(*self.lists):
            self.tree.insert("", "end", values=item)

        # adjust each column's width by maximum length string:
        for i, col in enumerate(self.headers):
            try:
                s = max(self.lists[i], key=len)
            except TypeError:
                s = str(max(self.lists[i]))
            col_w = tkFont.Font().measure(s)
            width = self.tree.column(col, width=None)
            if width < col_w:
                self.tree.column(col, width=col_w)
        self.tree.bind("<Double-1>", self._double_click)


def _sortby(tree, col, descending):  # pragma: no cover
    """sort tree contents when a column header is clicked on"""
    # grab values to sort
    data = [(tree.set(child, col), child) for child in tree.get_children("")]

    # print(data)
    data.sort(reverse=descending)
    for ix, item in enumerate(data):
        tree.move(item[1], "", ix)

    # to make first of any current selection visible:
    tree.see(tree.selection()[:1])

    # switch the heading so it will sort in the opposite direction
    tree.heading(col, command=lambda col=col: _sortby(tree, col, int(not descending)))
