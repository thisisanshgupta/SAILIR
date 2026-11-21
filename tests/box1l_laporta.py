"""Small bounded finite-field Laporta reference solver used by box tests."""
import itertools

P = 1009

def weight(a):
    return (sum(max(x, 0) for x in a), -sum(min(x, 0) for x in a), tuple(abs(x) for x in a))

def solve_box(env):
    """Solve all IBPs for seeds with positive weight <=5 and numerator <=2."""
    top = env.TOPOLOGY
    templates = env.build_box_templates()
    masters = set(top.masters)
    seeds = [a for a in itertools.product(range(-2, 6), repeat=4)
             if weight(a)[0] <= 5 and weight(a)[1] <= 2 and not env.is_zero(a)]
    equations = []
    for seed in seeds:
        for op in range(top.n_templates):
            eq = env.get_raw_equation(templates, {}, op, seed)
            if eq: equations.append(eq)
    variables = {i for row in equations for i in row}
    ordered = sorted((i for i in variables if i not in masters),
                     key=lambda i: (-weight(i)[0], -weight(i)[1], weight(i)[2]))
    ordered += sorted(masters)
    col = {i: j for j, i in enumerate(ordered)}

    # Sparse forward Gaussian elimination; masters remain free columns.
    pivots = {}
    for equation in equations:
        row = {col[i]: value % P for i, value in equation.items() if value % P}
        while row:
            pivot = min(row)
            if ordered[pivot] in masters: break
            value = row[pivot]
            if pivot in pivots:
                for j, coeff in pivots[pivot].items():
                    row[j] = (row.get(j, 0) - value * coeff) % P
                row = {j: x for j, x in row.items() if x}
            else:
                inv = pow(value, P - 2, P)
                pivots[pivot] = {j: x * inv % P for j, x in row.items()}
                break

    def expand(index, stack=()):
        integral = ordered[index]
        if integral in masters or index not in pivots:
            return {integral: 1}
        if index in stack: raise RuntimeError('cyclic Laporta pivot dependency')
        result = {}
        for j, coefficient in pivots[index].items():
            if j == index: continue
            for master, value in expand(j, stack + (index,)).items():
                result[master] = (result.get(master, 0) - coefficient * value) % P
        return {i: c for i, c in result.items() if c}

    def reduce(integral):
        if integral not in col: raise KeyError(f'integral outside Laporta basis: {integral}')
        return expand(col[integral])
    return reduce
