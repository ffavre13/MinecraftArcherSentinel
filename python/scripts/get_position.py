import numpy as np
from mcbot import McBotError, MinecraftClient
from scripts.constantes import CLASS_ENTITIES, DEFAULT_VY, PLAYER_CAM_HEIGHT
from scripts.utils import rotation_y


def get_position(mc):
    # List all entities
    entities = mc.get_entities(radius=16, types=CLASS_ENTITIES, limit=5)

    if len(entities) == 0:
        print("[LOG] - No entities detected")
        return

    selected_entity = np.random.choice(entities)

    # Entity coordinate
    entity_pos = np.array(selected_entity.pos)

    z_axe = np.array([0, 0, 1])
    alpha = np.deg2rad(np.random.randint(-180, 181))
    distance = np.random.randint(1, 26)
    height = np.array([0, np.random.randint(0, 11), 0])

    # Calculate the new direction
    z_rotated = np.matmul(rotation_y(-alpha), z_axe)
    new_coordinate = entity_pos + distance * z_rotated
    new_coordinate = new_coordinate + height

    # Travael to the coordinate
    try:
        info = mc.fly_to(*new_coordinate, max_steps=40)
    except McBotError:
        pass

    _, info = mc.step(0)

    # Wait until the player stop moving
    vx, vy, vz = info["motion"]
    while vx != 0 or (vy != 0 and vy != DEFAULT_VY) or vz != 0:
        _, info = mc.step(10)
        vx, vy, vz = info["motion"]

    mc.turn(-info["yaw"], 1)
    _, info = mc.look(-info["pitch"], 1)

    # Rotate in the entity direction
    player_pos = np.array(info["pos"]) + np.array([0, PLAYER_CAM_HEIGHT, 0])
    player_entity = entity_pos - player_pos

    # YAW
    # Projection on the xz plan
    player_entity_yaw = np.array([player_entity[0], 0, player_entity[2]])

    yaw = np.arccos(
        np.dot(player_entity_yaw, z_axe)
        / (np.linalg.norm(player_entity_yaw) * np.linalg.norm(z_axe))
    )
    yaw = np.rad2deg(yaw)

    if player_pos[0] < entity_pos[0]:
        yaw = -yaw
    mc.turn(yaw, 1)

    # PITCH
    player_entity_pitch = player_entity

    pitch = np.arccos(
        np.dot(player_entity_pitch, player_entity_yaw)
        / (np.linalg.norm(player_entity_pitch) * np.linalg.norm(player_entity_yaw))
    )
    pitch = np.rad2deg(pitch)

    if player_pos[1] < entity_pos[1]:
        pitch = -pitch
    mc.look(pitch, 1)

    # Randomly move the gaze
    random_yaw = np.random.randint(-35, 36)
    random_pitch = np.random.randint(-15, 41)
    mc.turn(random_yaw, 1)
    mc.look(random_pitch, 1)


if __name__ == "__main__":
    with MinecraftClient() as mc:
        get_position(mc)
