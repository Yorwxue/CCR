import numpy as np
import cv2
from PIL import Image
import json


class boxes_class(object):
    def __init__(self):
        self.boxes = list()
        self.number = 0

    def get_boxes(self, boxes):
        self.boxes = boxes
        self.number = len(boxes)

    def boxes_to_labelme(self, filepath, texts):
        with open(filepath.split('.')[0]+".json", 'w') as fw:
            fw.write('{\n')
            fw.write('  "flags": {},\n')
            fw.write('  "fillColor": [\n    255,\n    0,\n    0,\n    128\n  ],\n')
            fw.write('  "lineColor": [\n    0,\n    255,\n    0,\n    128\n  ],\n')
            fw.write('  "imagePath": "%s",\n' % (filepath.split('/')[-1]))
            fw.write('  "imageData": null,\n')
            fw.write('  "shapes": [\n')
            for idx, box in enumerate(self.boxes):
                fw.write('    {\n')
                fw.write('      "points": [\n')
                for point_idx, box_point in enumerate(box):
                    if point_idx != len(box)-1:
                        fw.write('        [\n          %d,\n          %d\n        ],\n' % (box_point[0], box_point[1]))
                    else:
                        fw.write('        [\n          %d,\n          %d\n        ]\n' % (box_point[0], box_point[1]))
                fw.write('      ],\n')
                fw.write('      "line_color": null,\n')
                fw.write('      "label": "%s",\n' % (texts))
                fw.write('      "fill_color": null\n')
                if idx != len(self.boxes)-1:
                    fw.write('    },\n')
                else:
                    fw.write('    }\n')
            fw.write('  ]\n')
            fw.write('}')
            print("save: %s" % filepath.split('.')[0]+".json")

