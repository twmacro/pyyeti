# turn off plotting for pytest:
try:
    import matplotlib as mpl
except ImportError:
    pass
else:
    mpl.interactive(False)
    mpl.use("Agg")
