import numpy as np
import json

#with open("/exp/scale18/ocr/users/srawls/slam_farsi_traindevtest/desc.json", 'r') as fh:
#    data = json.load(fh)

with open("/exp/scale18/ocr/users/srawls/slam_tamil_traindevtest/desc.json", 'r') as fh:
    data = json.load(fh)

#with open("/exp/scale18/ocr/users/srawls/yomdle_farsi_traindevtest/desc.json", 'r') as fh:
#    data = json.load(fh)

#with open("/exp/scale18/ocr/users/srawls/yomdle_tamil_traindevtest/desc.json", 'r') as fh:
#    data = json.load(fh)


heights = []
nwidths = []
for entry in data['train']:
    h = entry['height']
    w = entry['width']
    nh = 30
    nw = w * (nh / h)

    heights.append(h)
    nwidths.append(nw)


print("Avg height = %f" % np.mean(heights))
print("Avg normalized width = %f" % np.mean(nwidths))
