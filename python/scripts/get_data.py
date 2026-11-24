import os

import cv2
import numpy as np
from mcbot import MinecraftClient
from PIL import Image
from scripts.constantes import (
    CLASS_ENTITIES,
    FOV,
    GRID_SIZE_PX,
    HEIGHT,
    IMAGES_PATH,
    MARGIN,
    PLAYER_CAM_HEIGHT,
    WIDTH,
)


class hitbox:
    def __init__(
        self, center, width, height, points=None, distance=None, entityClass=None
    ):
        self.screen_points = None
        self.bounding_box = None  # bx, by, bw, bh
        self.distance = distance
        self.entityClass = entityClass

        if not points:
            points = []

            points.append(center + np.array([width / 2, 0, width / 2]))
            points.append(center + np.array([-width / 2, 0, width / 2]))
            points.append(center + np.array([-width / 2, 0, -width / 2]))
            points.append(center + np.array([width / 2, 0, -width / 2]))

            points.append(center + np.array([width / 2, height, width / 2]))
            points.append(center + np.array([-width / 2, height, width / 2]))
            points.append(center + np.array([-width / 2, height, -width / 2]))
            points.append(center + np.array([width / 2, height, -width / 2]))

        self.points = points

    def __call__(self, C, yaw, pitch):
        self.translation(C)
        self.rotation(yaw, pitch)
        self.transform_3D_to_2D()
        self.get_bounding_box()

    def translation(self, C):
        self.points = [point - C for point in self.points]

    def rotation(self, yaw, pitch):
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(-pitch)

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)

        Ry = np.array([[cos_yaw, 0, sin_yaw], [0, 1, 0], [-sin_yaw, 0, cos_yaw]])
        Rx = np.array(
            [[1, 0, 0], [0, cos_pitch, -sin_pitch], [0, sin_pitch, cos_pitch]]
        )

        self.points = [np.linalg.matmul(Ry, point) for point in self.points]
        self.points = [np.linalg.matmul(Rx, point) for point in self.points]

    def transform_3D_to_2D(self):
        screen_distance = HEIGHT / (2 * np.tan(np.deg2rad(FOV) / 2))
        self.screen_points = [
            np.array(
                [
                    int(WIDTH / 2 - point[0] * screen_distance / point[2]),
                    int(HEIGHT / 2 - point[1] * screen_distance / point[2]),
                ]
            )
            for point in self.points
        ]

    def get_bounding_box(self):
        x_min = max(0, min([point[0] for point in self.screen_points]))
        x_max = min(WIDTH - 1, max([point[0] for point in self.screen_points]))
        y_min = max(0, min([point[1] for point in self.screen_points]))
        y_max = min(HEIGHT - 1, max([point[1] for point in self.screen_points]))

        bx = int(x_min + (x_max - x_min) / 2)
        by = int(y_min + (y_max - y_min) / 2)
        bw = int(x_max - x_min)
        bh = int(y_max - y_min)

        self.bounding_box = np.array([bx, by, bw, bh])


def display_hitbox_points(image, hitboxes, path):
    os.makedirs(path, exist_ok=True)

    hitboxes_image = image.copy()
    for hb in hitboxes:
        for point in hb.screen_points:
            if hitboxes_image is not None:
                hitboxes_image = cv2.circle(
                    hitboxes_image, point, radius=0, color=(255, 0, 0), thickness=-1
                )

    if hitboxes_image is not None:
        Image.fromarray(hitboxes_image).save(path + "hitboxes.png")


def display_bounding_box(image, hitboxes, path):
    os.makedirs(path, exist_ok=True)

    bounding_box_image = image.copy()
    for hb in hitboxes:
        bx, by, bw, bh = hb.bounding_box
        bounding_box_image = cv2.rectangle(
            bounding_box_image,
            (bx - bw // 2, by - bh // 2),
            (bx + bw // 2, by + bh // 2),
            (255, 0, 0),
            1,
        )

    if bounding_box_image is not None:
        Image.fromarray(bounding_box_image).save(path + "bounding_box.png")


def save_data(hitboxes, path):
    os.makedirs(path, exist_ok=True)

    data = {
        "hitboxes_points": [hb.screen_points for hb in hitboxes],
        "bounding_boxes": [hb.bounding_box for hb in hitboxes],
    }

    np.savez(path + "data.npz", data)


def save_label(hitboxes, path):
    os.makedirs(path, exist_ok=True)

    available_hitboxes = [hb for hb in hitboxes]
    grids = []

    for x in range(0, WIDTH, GRID_SIZE_PX):
        tmp_grid_line = []
        for y in range(0, HEIGHT, GRID_SIZE_PX):
            possible_candidates = []

            for hb in available_hitboxes:
                if (
                    hb.bounding_box[0] >= x
                    and hb.bounding_box[0] < x + GRID_SIZE_PX
                    and hb.bounding_box[1] >= y
                    and hb.bounding_box[1] < y + GRID_SIZE_PX
                ):
                    possible_candidates.append(hb)

            if possible_candidates:
                possible_candidates = sorted(
                    possible_candidates, key=lambda hb: hb.distance
                )

                current_hitbox = possible_candidates[0]
                available_hitboxes.remove(current_hitbox)
                label = np.array(
                    [
                        1,
                        current_hitbox.bounding_box[0],
                        current_hitbox.bounding_box[1],
                        current_hitbox.bounding_box[2],
                        current_hitbox.bounding_box[3],
                        1 if current_hitbox.entityClass == CLASS_ENTITIES[0] else 0,
                        1 if current_hitbox.entityClass == CLASS_ENTITIES[1] else 0,
                        1 if current_hitbox.entityClass == CLASS_ENTITIES[2] else 0,
                    ]
                )
            else:
                label = np.array([0, 0, 0, 0, 0, 0, 0, 0])

            tmp_grid_line.append(label)

        grids.append(tmp_grid_line)

    np.savez(path + "label.npz", grids)


def main():
    with MinecraftClient() as mc:
        info = mc.wait_for_world()
        mc.config(width=WIDTH, height=HEIGHT)
        for i in range(10):
            # Wait 2 second between capture
            _, info = mc.step(40)

            print("[WAITING] - player should not move")

            # Capture only if the player is not moving
            vx, vy, vz = info["motion"]
            default_vy = -0.0784000015258789
            while vx != 0 or (vy != 0 and vy != default_vy) or vz != 0:
                _, info = mc.step(40)
                vx, vy, vz = info["motion"]

            print("[CAPTURE] - image and entities")

            image, info = mc.step(
                ticks=1,
                entities={"types": CLASS_ENTITIES, "visibility": True, "radius": 50},
                frame=True,
            )

            print(
                "Number of detected entities:",
                np.sum(
                    [
                        1
                        for e in info["entities"]
                        if e.in_view and e.line_of_sight and not e.invisible
                    ]
                ),
            )

            if image is not None:
                path = IMAGES_PATH + f"{i}/"
                os.makedirs(path, exist_ok=True)
                Image.fromarray(image).save(path + "before.png")

            C = np.array(info["pos"]) + np.array([0, PLAYER_CAM_HEIGHT, 0])
            yaw = info["yaw"]
            pitch = info["pitch"]
            hitboxes = []

            for e in info["entities"]:
                if e.in_view and e.line_of_sight and not e.invisible:
                    current_hitbox = hitbox(
                        np.array(e.pos), e.size[0], e.size[1], None, e.distance, e.type
                    )
                    current_hitbox(C, yaw, pitch)

                    if (
                        current_hitbox.bounding_box[2] > MARGIN
                        or current_hitbox.bounding_box[2] > MARGIN
                    ):
                        hitboxes.append(current_hitbox)

            # Display hitbox points
            display_hitbox_points(image, hitboxes, IMAGES_PATH + f"{i}/")

            # Display bounding box
            display_bounding_box(image, hitboxes, IMAGES_PATH + f"{i}/")

            # Save data in a npz file
            save_data(hitboxes, IMAGES_PATH + f"{i}/")

            # Save label in a npz file
            save_label(hitboxes, IMAGES_PATH + f"{i}/")


if __name__ == "__main__":
    main()
