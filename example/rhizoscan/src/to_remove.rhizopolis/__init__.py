
# Redirect path
import os

cdir = os.path.dirname(__file__)
pdir = os.path.join(cdir, "../rhizoscan")
pdir = os.path.abspath(pdir)

__path__ = [pdir] + __path__[:]

from rhizoscan.__init__ import *
