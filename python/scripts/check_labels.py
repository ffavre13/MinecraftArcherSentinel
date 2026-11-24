import numpy as np

label = np.load("frames/data/0/label.npz")
label = label["arr_0"]

for y in range(len(label[0])):
    s = ""
    for x in range(len(label)):
        s += str(label[y,x,0])
    print(s)