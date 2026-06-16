# PyInstaller hook for astropy — override of the pyinstaller-hooks-contrib hook.
#
# The stock contrib hook does an unfiltered ``collect_submodules('astropy')``,
# which recurses into ``astropy.visualization.wcsaxes``. That module's __init__
# calls ``pytest.importorskip("matplotlib")`` at import time; with matplotlib
# absent (it is not a dependency of oeuvre), importorskip raises
# ``pytest.outcomes.Skipped`` — a *BaseException*, so collect_submodules'
# ``on_error`` handling (which only catches ``Exception``) can't swallow it and
# the whole build aborts.
#
# This override is identical to the contrib hook except it filters the
# ``wcsaxes`` subtree out of submodule collection so it is never imported.
# oeuvre only uses ``astropy.io.fits``, so dropping the matplotlib-only WCS
# plotting helpers costs us nothing.
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
    is_module_satisfies,
)

datas = collect_data_files('astropy')

hiddenimports = collect_submodules(
    'astropy',
    filter=lambda name: not name.startswith('astropy.visualization.wcsaxes'),
)

# *_parsetab.py / *_lextab.py are loaded as files, not imported as submodules.
ply_files = []
for path, target in collect_data_files('astropy', include_py_files=True):
    if path.endswith(('_parsetab.py', '_lextab.py')):
        ply_files.append((path, target))
datas += ply_files

if is_module_satisfies('astropy >= 5.0'):
    datas += copy_metadata('astropy')
    datas += copy_metadata('numpy')

hiddenimports += ['numpy.lib.recfunctions']
