import importlib.util
import random
import sys
from pathlib import Path
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    import types
    sys.modules['torch'] = types.ModuleType('torch')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sailir'))
spec = importlib.util.spec_from_file_location('boxgen', ROOT / 'scripts/data_gen/generate_multisector_data.py')
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)

def test_200_box_scrambles_unscramble_to_corner():
    gen.TOPOLOGY = gen.get_topology('box1l')
    gen.shared_env.set_topology('box1l'); gen.shared_env.set_prime(1009); gen.PRIME = 1009
    ibp = gen.parse_templates('ignored'); li = {}
    shifts = {op: [shift for shift, _ in terms] for op, terms in ibp.items()}
    rng = random.Random(8128)
    sectors = gen.get_all_valid_sectors()
    for scramble_id in range(200):
        sector = rng.choice(sectors)
        corner = gen.get_corner_integral(sector)
        coefficient = rng.randrange(1, 1009)
        original = {corner: coefficient}
        gen.random.seed(rng.randrange(2**31))
        scrambled, actions = gen.scramble(original, ibp, li, 4, rng.randint(5, 20), sector)

        expr, subs, used = dict(scrambled), {}, set()
        for _ in range(500):
            sector_expr = gen.filter_sector_only(expr, sector)
            nonmasters = {i: c for i, c in sector_expr.items() if not gen.is_master_for_sector(i, sector)}
            if not nonmasters: break
            target = sorted(nonmasters, key=lambda x: (-gen.weight(x)[0], -gen.weight(x)[1], gen.weight(x)[2]))[0]
            chosen = None
            for idx, (op, seed) in enumerate(actions):
                if idx in used: continue
                raw = gen.get_raw_equation(ibp, li, op, seed)
                reduced = gen.apply_all_substitutions(raw, subs)
                if reduced.get(target, 0): chosen = (idx, op, seed, reduced); break
            assert chosen is not None, (scramble_id, target, actions)
            idx, op, seed, reduced = chosen
            delta = tuple(seed[i] - target[i] for i in range(gen.TOPOLOGY.nidx))
            valid = gen.enumerate_valid_actions(target, subs, ibp, li, shifts, sector)
            assert (op, delta) in valid
            solution = gen.solve_ibp_for(reduced, target)
            assert solution is not None
            used.add(idx); subs[target] = solution
            expr = gen.apply_substitution(expr, target, solution)
        else:
            raise AssertionError(f'scramble {scramble_id} did not terminate')
        final = {i: c for i, c in gen.filter_sector_only(expr, sector).items()
                 if gen.is_master_for_sector(i, sector)}
        assert final == original, (scramble_id, final, original)
