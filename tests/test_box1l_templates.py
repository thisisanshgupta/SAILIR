import sys
from pathlib import Path
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    # ibp_env only needs torch for its model-input helper; algebra tests are pure Python.
    import types
    sys.modules['torch'] = types.ModuleType('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sailir'))
import ibp_env as env

def setup_module():
    env.set_topology('box1l')
    env.set_prime(1009)
    env.KINEMATICS = {'d': 41, 's': 17, 't': 23}

def test_box_templates_and_vanishing_terms():
    e = env.IBPEnvironment()
    assert e.get_raw_equation_cached(0, (1,-1,1,0)) == {(1,-1,1,0):39, (1,-1,2,0):17}
    assert e.get_raw_equation_cached(2, (1,-1,1,0)) == {(1,-1,1,0):39, (2,-1,1,0):17}
    # Scaleless integrals are dropped from the equation.
    assert e.get_raw_equation_cached(1, (1,0,1,0)) == {(1,0,1,0):39, (2,-1,1,0):1008, (1,-1,2,0):1008}
    assert e.get_raw_equation_cached(2, (1,0,1,0)) == {(1,0,1,0):38, (2,0,1,0):17}

def test_vanishing_integrals_are_filtered_and_eval_errors_raise():
    assert env.is_zero((1,0,0,1))
    assert not env.is_zero((1,-1,1,0))
    try:
        env.eval_coeff('__import__("os").system("true")', (1,0,1,0))
    except ValueError:
        pass
    else:
        raise AssertionError('restricted coefficient evaluator accepted an unsafe expression')

def test_box_goal_states_and_masters_only_mode():
    from topology import get_topology
    topo = get_topology('box1l')
    assert len(topo.goal_states()) == 7
    env.set_paper_masters_only(False)
    assert env.is_master((1,1,1,0))  # training corner of sector 1110
    assert env.is_master((1,0,1,0))  # true master
    env.set_paper_masters_only(True)
    assert not env.is_master((1,1,1,0))
    assert env.is_master((1,0,1,0))
    env.set_paper_masters_only(False)
