from mcbot import MinecraftClient
from PIL import Image
from scripts.constantes import HEIGHT, WIDTH

with MinecraftClient() as mc:
    info = mc.wait_for_world()
    mc.config(width=WIDTH, height=HEIGHT, hide_gui=False)
    image, _ = mc.step(40, frame=True)
    Image.fromarray(image).save("frames/forest.png")
