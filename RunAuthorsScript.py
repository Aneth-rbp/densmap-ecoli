"""Runs an authors script unchanged on numba 0.50 (aarch64): only import and compile-flag shims."""
import runpy, sys
import numba, numba.core.errors
sys.modules["numba.errors"] = numba.core.errors
import densmap.densmap_ as dm
# parallel=True miscompiles on numba 0.50 aarch64 (ZeroDivisionError, NaN); the function has no prange
dm.optimize_layout = numba.njit(fastmath=True)(dm.optimize_layout.py_func)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
