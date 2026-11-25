import numpy as np


def check_label(label):
    for y in range(len(label)):
        s = ""
        for x in range(len(label[0])):
            s += str(label[y, x, 0])
        print(s)


if __name__ == "__main__":
    label = np.load("frames/data/47/label.npy")
    print(label.shape)
    check_label(label)
