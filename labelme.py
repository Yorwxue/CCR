import numpy as np
import cv2
from PIL import Image
import json

from eval import OCD, OCR


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
                fw.write('      "label": "%s",\n' % (texts[idx]["text"]))
                fw.write('      "fill_color": null\n')
                if idx != len(self.boxes)-1:
                    fw.write('    },\n')
                else:
                    fw.write('    }\n')
            fw.write('  ]\n')
            fw.write('}')


if __name__ == '__main__':
    boxes_obj = boxes_class()

    ocd = OCD("/home/c11tch/workspace/PycharmProjects/JC_Demo/ocr_module/EAST/pretrained_model/east_mixed_149482/")
    # ocd = OCD("/data2/relabelled/east_icdar2015_resnet_v1_50_rbox")
    ocr = OCR()

    image_path = "/home/c11tch/Pictures/ocd_test_img/aPICT0034.JPG"
    image = cv2.imread(image_path)

    # ocd
    image_list, masked_image, boxes = ocd.detection(image)

    # ocr
    result_list = ocr.recognize(image_list)
    for i in result_list:
        print(i)

    # labelme
    # boxes_obj.get_boxes(boxes)
    # boxes_obj.boxes_to_labelme(image_path, result_list)
    pass
