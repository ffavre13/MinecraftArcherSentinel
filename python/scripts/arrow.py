import numpy as np
from scripts.utils import rotation_x, rotation_y

def velocity_t(tick, velocity_0, drag, gravity):
    return drag**tick * velocity_0 - gravity * (1 - drag**tick) / (1 - drag)

def pos_t(tick, pos_0, velocity_0, drag, gravity):
    return pos_0 + (1-drag**tick) / (1- drag) * velocity_0 - gravity * (tick - (1-drag**tick) / (1- drag)) / (1 - drag)

# https://minecraft.wiki/w/Arrow
# https://en.wikipedia.org/wiki/Geometric_series
def calculate_angle(target_pos):
    current_pos = np.array([0, 0, 0])
    norm_velocity_0 = 3
    drag = 0.99
    gravity = np.array([0,0.05,0])
    velocity_0 = np.array([0,0,3])
    print(velocity_t(30, velocity_0, drag, gravity))
    print(pos_t(30, current_pos, velocity_0, drag, gravity))

if __name__ == "__main__":
    print(calculate_angle(np.array([10, 0, 0])))
             