import cv2
import numpy as np
from mcbot import MinecraftClient
from PIL import Image
from scripts.constantes import (
    CLASS_ENTITIES,
    FOV,
    HEIGHT,
    IMAGES_PATH,
    PLAYER_CAM_HEIGHT,
    WIDTH,
)


class hitbox:
    def __init__(self, center, width, height, points=None):
        self.screen_points = None

        if not points:
            points = []

            points.append(center + np.array([width/2, 0, width/2]))
            points.append(center + np.array([-width/2, 0, width/2]))
            points.append(center + np.array([-width/2, 0, -width/2]))
            points.append(center + np.array([width/2, 0, -width/2]))

            points.append(center + np.array([width/2, height, width/2]))
            points.append(center + np.array([-width/2, height, width/2]))
            points.append(center + np.array([-width/2, height, -width/2]))
            points.append(center + np.array([width/2, height, -width/2]))

        self.points = points

    def __call__(self, C, yaw, pitch):
        self.translation(C)
        self.rotation(yaw, pitch)
        self.transform_3D_to_2D()

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


def main():
    with MinecraftClient() as mc:
        mc.config(width=WIDTH, height=HEIGHT)
        info = mc.wait_for_world()
        for i in range(1):
            image, info = mc.step(
                ticks=20,
                entities={"types": CLASS_ENTITIES, "visibility": True},
                frame=True,
            )

            if image is not None:
                Image.fromarray(image).save(IMAGES_PATH + "before.png")

            image = image.copy()
            C = np.array(info["pos"]) + np.array([0, PLAYER_CAM_HEIGHT, 0])
            yaw = info["yaw"]
            pitch = info["pitch"]
            hitboxes = []

            for e in info["entities"]:
                if e.in_view and e.line_of_sight:
                    current_hitbox = hitbox(np.array(e.pos), e.size[0], e.size[1])
                    current_hitbox(C, yaw, pitch)
                    hitboxes.append(current_hitbox)

                    for point in current_hitbox.screen_points:
                        if image is not None:
                            image = cv2.circle(
                                image, point, radius=0, color=(255, 0, 0), thickness=-1
                            )

            if image is not None:
                Image.fromarray(image).save(IMAGES_PATH + "after.png")


if __name__ == "__main__":
    main()
