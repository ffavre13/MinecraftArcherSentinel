import numpy as np
from scripts.get_data import hitbox


def test_translation():
    test_hitbox = hitbox(np.array([0, 0, 0]), 0, 0, [np.array([10, 10, 10])])
    center_point = np.array([10, 10, 10])
    test_hitbox.translation(center_point)
    assert all(a == b for a, b in zip(test_hitbox.points[0], np.array([0, 0, 0])))

def test_rotation_yaw_90():
    test_hitbox = hitbox(np.array([0, 0, 0]), 0, 0, [np.array([0, 0, 1])])
    yaw = 90
    test_hitbox.rotation(yaw, 0)
    print(test_hitbox.points[0])
    assert all(int(a) == b for a, b in zip(test_hitbox.points[0], np.array([1, 0, 0])))

def test_rotation_yaw_180():
    test_hitbox = hitbox(np.array([0, 0, 0]), 0, 0, [np.array([0, 0, 1])])
    yaw = 180
    test_hitbox.rotation(yaw, 0)
    print(test_hitbox.points[0])
    assert all(int(a) == b for a, b in zip(test_hitbox.points[0], np.array([0, 0, -1])))

def test_rotation_yaw_minus_90():
    test_hitbox = hitbox(np.array([0, 0, 0]), 0, 0, [np.array([0, 0, 1])])
    yaw = -90
    test_hitbox.rotation(yaw, 0)
    print(test_hitbox.points[0])
    assert all(int(a) == b for a, b in zip(test_hitbox.points[0], np.array([-1, 0, 0])))

def test_rotation_yaw_minus_180():
    test_hitbox = hitbox(np.array([0, 0, 0]), 0, 0, [np.array([0, 0, 1])])
    yaw = -180
    test_hitbox.rotation(yaw, 0)
    print(test_hitbox.points[0])
    assert all(int(a) == b for a, b in zip(test_hitbox.points[0], np.array([0, 0, -1])))

def test_rotation_pitch_90():
    test_hitbox = hitbox(np.array([0, 0, 0]), 0, 0, [np.array([0, 0, 1])])
    pitch = 90
    test_hitbox.rotation(0, pitch)
    print(test_hitbox.points[0])
    assert all(int(a) == b for a, b in zip(test_hitbox.points[0], np.array([0, 1, 0])))

def test_rotation_pitch_minus_90():
    test_hitbox = hitbox(np.array([0, 0, 0]), 0, 0, [np.array([0, 0, 1])])
    pitch = -90
    test_hitbox.rotation(0, pitch)
    print(test_hitbox.points[0])
    assert all(int(a) == b for a, b in zip(test_hitbox.points[0], np.array([0, -1, 0])))