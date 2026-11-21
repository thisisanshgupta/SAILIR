import sys
from pathlib import Path
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    import types
    sys.modules['torch'] = types.ModuleType('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sailir'))
import ibp_env as env
from topology import get_topology

def test_box_templates_preserve_seed_sector_or_subsector():
    env.set_topology('box1l')
    env.set_prime(1009)
    ibp = env.IBPEnvironment()
    topo = get_topology('box1l')
    for sector_id in topo.nonzero_sectors:
        corner = tuple((sector_id >> i) & 1 for i in range(topo.np))
        seeds = [corner]
        # Exercise positive powers, numerator powers, and absent propagators.
        for offset in range(4):
            seed = tuple((1 + offset if x else (-1 if (i + offset) % 3 == 0 else 0))
                         for i, x in enumerate(corner))
            seeds.append(seed)
        for seed in seeds:
            seed_mask = env.get_sector(seed)
            for op in range(topo.n_templates):
                equation = ibp.get_raw_equation_cached(op, seed)
                for integral in equation:
                    result_mask = env.get_sector(integral)
                    assert all(not result_mask[i] or seed_mask[i] for i in range(topo.np)), (seed, op, integral)
