import cv2
import numpy as np
import torch
from PIL import Image
from scripts.check_labels import check_label
from scripts.utils import CNN

if __name__ == "__main__":
    model = CNN()
    model.load_state_dict(torch.load("models/target_detector.pt", weights_only=True))
    model.eval()
    input_image = np.array(Image.open("frames/data/0/before.png"))

    input_tensor = torch.FloatTensor(input_image)
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output).numpy()[0]
        for y in range(len(output)):
            for x in range(len(output[0])):
                output[y, x, 0] = 0 if output[y, x, 0] < 0.5 else 1

        print(output.shape)
        check_label(output)

    bounding_box_image = input_image.copy()

    for y in range(len(output)):
        for x in range(len(output[0])):
            if output[y, x, 0] == 1:
                cv2.rectangle(
                    bounding_box_image,
                    (
                        int(output[y, x, 1] - output[y, x, 3] // 2),
                        int(output[y, x, 2] - output[y, x, 4] // 2),
                    ),
                    (
                        int(output[y, x, 1] + output[y, x, 3] // 2),
                        int(output[y, x, 2] + output[y, x, 4] // 2),
                    ),
                    (255, 0, 0),
                    1,
                )

    Image.fromarray(bounding_box_image).save("frames/tests/bounding_box.png")
