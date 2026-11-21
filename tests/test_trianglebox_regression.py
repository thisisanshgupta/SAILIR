import sys
from pathlib import Path
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    import types
    sys.modules['torch'] = types.ModuleType('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sailir'))
import ibp_env as env

def test_default_trianglebox_raw_equation_is_unchanged():
    env.set_topology('trianglebox')
    env.set_prime(1009)
    environment = env.IBPEnvironment()
    seed = (1,1,1,1,1,1,0)
    assert environment.get_raw_equation_cached(0, seed) == {
        seed: 37,
        (0,1,2,1,1,1,0): 1008,
        (1,0,2,1,1,1,0): 1,
        (0,1,1,2,1,1,0): 1008,
        (1,1,1,2,1,1,0): 1,
    }
    assert environment.get_raw_equation_cached(0, (1,0,1,0,1,1,0)) == {
        (1,0,1,0,1,1,0): 38,
        (0,0,2,0,1,1,0): 1008,
        (1,-1,2,0,1,1,0): 1,
    }
    assert environment.get_raw_equation_cached(0, (1,1,1,1,1,1,1)) == {
        (1,1,1,1,1,1,1): 36,
        (0,1,2,1,1,1,1): 1008,
        (1,0,2,1,1,1,1): 1,
        (1,1,1,1,1,1,2): 47,
        (0,1,1,2,1,1,1): 1008,
        (0,1,1,1,1,1,2): 1008,
        (1,1,1,2,1,1,1): 1,
    }
