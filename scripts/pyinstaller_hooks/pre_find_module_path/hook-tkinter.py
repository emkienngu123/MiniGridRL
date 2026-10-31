"""Keep Tkinter discoverable when the builder cannot initialize Tcl.

The Windows build script validates and explicitly collects matching Tcl/Tk
scripts. Creating an interpreter at build time is unnecessary and can fail in
restricted build runners despite a complete, working Python installation.
"""


def pre_find_module_path(hook_api):
    pass
