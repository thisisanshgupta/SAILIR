import sys
from pathlib import Path
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
