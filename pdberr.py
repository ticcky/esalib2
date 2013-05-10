# encoding: utf8
# author: http://stackoverflow.com/users/746961/constantinius
import sys

orig_excepthook = None

def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, ipdb
        # we are NOT in interactive mode, print the exception…
        traceback.print_exception(type, value, tb)
        print
        # …then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        ipdb.post_mortem(tb) # more “modern”

def init():
        global orig_excepthook
        orig_excepthook = sys.excepthook
        sys.excepthook = info

def deinit():
        global orig_excepthook
        if orig_excepthook is not None:
                sys.excepthook = sys.excepthook
