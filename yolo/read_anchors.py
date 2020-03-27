
import numpy as np

# read anchors from file
file = "./data/voc_anchor.npy"
anchors = np.load(file)
print(anchors)
print(anchors.shape)

print(anchors.flatten())
