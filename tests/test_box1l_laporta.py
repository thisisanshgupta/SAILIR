import sys
from pathlib import Path
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    import types
    sys.modules['torch'] = types.ModuleType('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sailir'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ibp_env as env
from box1l_laporta import solve_box

def test_box_laporta_reference_reductions():
    env.set_topology('box1l')
    env.set_prime(1009)
    environment = env.IBPEnvironment()
    reduce = solve_box(env)
    assert reduce((1,-1,1,0)) == {(1,0,1,0):496}
    assert set(reduce((2,-2,2,1))) <= environment.topology.masters
    assert set(reduce((2,2,1,-1))) <= environment.topology.masters
