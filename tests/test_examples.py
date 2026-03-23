import os
import glob
import runpy
from unittest.mock import patch
import pytest

# Find all example scripts
examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))
example_files = glob.glob(os.path.join(examples_dir, '*.py'))

@pytest.mark.parametrize("script_path", sorted(example_files))
@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_example_script(mock_save, mock_show, script_path):
    # To speed up and avoid full 5000s setups hanging, we can patch RK4.step (if used)
    # inside gnc_toolkit. However, it's safer to just let it raise a custom early break.
    from gnc_toolkit.integrators.rk4 import RK4
    
    # Simple trick to execute just a few loops
    orig_step = RK4.step
    def fast_step(self, *args, **kwargs):
        orig_step(self, *args, **kwargs)
        raise StopIteration("Simulation early success")

    with patch.object(RK4, 'step', fast_step):
        try:
            runpy.run_path(script_path, run_name="__main__")
        except (StopIteration, Exception) as e:
            # If it's our early break or list index issues from breaking early, it's a pass
            if "early success" in str(e) or isinstance(e, StopIteration):
                 pass
            else:
                 # Check if it fails with actually wrong exceptions before loop
                 pytest.fail(f"Example {os.path.basename(script_path)} failed with {type(e).__name__}: {e}")
