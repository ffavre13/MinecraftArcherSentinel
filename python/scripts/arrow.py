import numpy as np
from scripts.utils import rotation_x, rotation_y


# https://en.wikipedia.org/wiki/Geometric_series
def velocity_t(tick, velocity_0, drag, gravity):
    return drag**tick * velocity_0 - gravity * (1 - drag**tick) / (1 - drag)


def pos_t(tick, velocity_0, drag, gravity):
    return (1 - drag**tick) / (1 - drag) * velocity_0 - gravity * (
        tick - (1 - drag**tick) / (1 - drag)
    ) / (1 - drag)


def angle_to_velocity(yaw, pitch, norm_velocity_0):
    yaw = np.deg2rad(-yaw)
    pitch = np.deg2rad(pitch)

    velocity = np.array([0, 0, 1])
    velocity = np.matmul(rotation_y(yaw), np.matmul(rotation_x(pitch), velocity))
    velocity = velocity / np.linalg.norm(velocity) * norm_velocity_0
    return velocity


# https://minecraft.wiki/w/Arrow
def calculate_angle(target_pos):
    norm_velocity_0 = 3
    drag = 0.99
    gravity = np.array([0, 0.05, 0])

    # Calculate the yaw
    vector_z = np.array([0, 0, 1])
    target_pos_xz = np.array([target_pos[0], 0, target_pos[2]])
    yaw = np.rad2deg(
        -np.acos(
            np.dot(vector_z, target_pos_xz)
            / (np.linalg.norm(vector_z) * np.linalg.norm(target_pos_xz))
        )
    )

    # TODO: Calculate the pitch

    velocity_0 = angle_to_velocity(yaw, 0, norm_velocity_0)
    print(velocity_0)
    print(velocity_t(30, velocity_0, drag, gravity))
    print(pos_t(30, velocity_0, drag, gravity))


if __name__ == "__main__":
    print(calculate_angle(np.array([10, 10, 10])))
