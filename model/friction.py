"""friction.py -- a slippery (or sticky) floor for humenv bodies.

MuJoCo takes a contact's friction from the geom with the higher priority, or the element-wise max when the priorities
are equal. The humanoid's feet are priority 1 with sliding friction 1.0 and every other body geom is priority 0 with
0.7, against a priority-0 floor at 0.7: changing the floor's friction alone changes nothing the feet touch, and lowering
it changes nothing at all (max). So the floor is given the highest priority, and then its friction is the contact's.
"""
import gymnasium


def set_floor_friction(m, mu, floor="floor"):
    """the floor wins every contact (priority above every geom) with sliding friction mu (torsional / rolling kept)."""
    g = m.geom(floor).id
    m.geom_priority[g] = int(m.geom_priority.max()) + 1
    m.geom_friction[g, 0] = mu
    return g


class FloorFrictionWrapper(gymnasium.Wrapper):
    """set_floor_friction on the env's own model, so it holds inside every worker of a vector env."""

    def __init__(self, env, mu, floor="floor"):
        super().__init__(env)
        set_floor_friction(env.unwrapped.model, mu, floor)
