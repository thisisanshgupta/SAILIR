"""Topology definitions shared by generation, reduction, and modeling."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Topology:
    name: str
    np: int
    n_isp: int
    n_templates: int
    masters: frozenset
    kinematics: dict
    nonzero_sectors: tuple | None = None
    top_sector: tuple | None = None
    @property
    def nidx(self): return self.np + self.n_isp
    @property
    def n_templates_total(self): return self.n_templates
    def is_zero(self, a): return is_zero_box(a) if self.name == 'box1l' else False
    def goal_states(self):
        if self.nonzero_sectors is not None:
            return frozenset(tuple((s >> i) & 1 for i in range(self.np)) + (0,) * self.n_isp for s in self.nonzero_sectors)
        masks_with_masters = {tuple(1 if a[i] > 0 else 0 for i in range(self.np)) for a in self.masters}
        corners = {tuple((s >> i) & 1 for i in range(self.np)) + (0,) * self.n_isp
                   for s in range(1, 1 << self.np)}
        corners = {a for a in corners if tuple(1 if x > 0 else 0 for x in a[:self.np]) not in masks_with_masters}
        return frozenset(set(self.masters) | corners)

TRIANGLEBOX_MASTERS = frozenset([
 (0,1,1,1,0,0,0),(1,0,1,0,1,0,0),(1,1,0,1,1,0,0),(0,0,1,1,1,0,0),
 (1,0,1,1,1,0,0),(1,-1,1,1,1,0,0),(0,1,1,1,1,0,0),(-1,1,1,1,1,0,0),
 (1,1,1,1,1,0,0),(1,0,1,0,0,1,0),(1,1,0,1,0,1,0),(1,0,1,0,1,1,0),
 (1,-1,1,0,1,1,0),(1,0,0,1,1,1,0),(1,1,0,1,1,1,0),(1,0,1,1,1,1,0)])
BOX_MASTERS = frozenset([(1,0,1,0),(0,1,0,1),(1,1,1,1)])
TOPOLOGIES = {
 'trianglebox': Topology('trianglebox',6,1,9,TRIANGLEBOX_MASTERS,{'d':41,'m1':1,'m2':31,'m3':47},None,(1,0,1,0,1,1)),
 'box1l': Topology('box1l',4,0,4,BOX_MASTERS,{'d':41,'s':17,'t':23},(5,10,7,14,13,11,15),None),
}
def get_topology(value='trianglebox'):
    if isinstance(value, Topology): return value
    try: return TOPOLOGIES[value]
    except KeyError: raise ValueError(f"Unknown topology {value!r}; choose from {', '.join(TOPOLOGIES)}")

def is_zero_box(a):
    return sum(x > 0 for x in a) < 2 or any(a[i] <= 0 and a[(i+1)%4] <= 0 for i in range(4))

def vanishes(topology, a):
    t=get_topology(topology)
    return t.is_zero(a)
