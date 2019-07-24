import cv2
import math
import numpy as np


def polyCrop(image, rect_box, poly_box):
    """
    Note: image must be uint8 type
    :param image:
    :param rect_box:
    :param poly_box:
    :return:
    """
    # rect_box: Minimum circumscribed rectangle　of ROI
    # poly_boxROI: polygon of ROI

    # original image
    # -1 loads as-is so if it will be 3 or 4 channel as the original
    display_img = image.copy()
    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)

    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array([poly_box], dtype=np.int32)

    # draw box
    point_polylines = np.array(poly_box, np.int32)
    point_polylines = point_polylines.reshape((-1, 1, 2))
    cv2.polylines(display_img, [point_polylines], True, (200, 150, 200), thickness=5)

    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)

    # crop
    max_x = rect_box[1][0]
    min_x = rect_box[0][0]
    max_y = rect_box[2][1]
    min_y = rect_box[0][1]
    crop_image = masked_image[min_y: max_y, min_x: max_x, :]

    return display_img, masked_image, crop_image


def ratioImputation(img, target_ration=(60, 180)):
    height = img.shape[0]
    width = img.shape[1]

    # base on height
    target_width = height / target_ration[0] * target_ration[1]
    target_width = math.floor(target_width) if int((target_width - int(target_width)) * 10) < 0.5 else math.ceil(target_width)  # 四捨五入

    # base on weight
    target_height = width / target_ration[1] * target_ration[0]
    target_height = math.floor(target_height) if int((target_height - int(target_height)) * 10) < 0.5 else math.ceil(target_height)  # 四捨五入

    if target_width > width:  # base on height
        new_img = np.zeros((height, target_width, 3), np.uint8)
        margin = target_width - width
        left_margin = int(margin / 2)
        # right = margin - left_margin
        ret_img = new_img.copy()
        ret_img[:, left_margin: (left_margin + width), :] = img

    else:  # base on weight
        new_img = np.zeros((target_height, width, 3), np.uint8)
        margin = target_height - height
        up_margin = int(margin / 2)
        # bottom_margin = margin - up_margin
        ret_img = new_img.copy()
        ret_img[up_margin: (up_margin + height), :, :] = img

    ret_img = cv2.resize(ret_img, (target_ration[1], target_ration[0]))
    return ret_img


if __name__ == "__main__":

    image = cv2.imread('../dataset/000.jpg', -1)
    # image = cv2.resize(image, None, fx=4, fy=4)

    # top_left, top_right, bottom_right, bottom_left
    point_polylines = [[60, 435], [533, 597], [1042, 278], [708, 216]]  # administrative no. 1

    max_x = np.max(np.asarray(point_polylines)[:, 0])
    min_x = np.min(np.asarray(point_polylines)[:, 0])
    max_y = np.max(np.asarray(point_polylines)[:, 1])
    min_y = np.min(np.asarray(point_polylines)[:, 1])
    point_rect = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

    display_img, masked_img, crop_image = polyCrop(image, rect_box=point_rect, poly_box=point_polylines)
    # ratio_img = ratioImputation(crop_image, target_ration=(60, 180))

    # save the result
    cv2.imwrite('../dataset/image_masked.png', masked_img)
    cv2.imwrite('../dataset/display_img.png', display_img)
    # cv2.imwrite('image_cropped.png', crop_image)
    # cv2.imwrite('ratio_img.png', ratio_img)

    # show
    # cv2.imshow("img_ROI", display_img)
    # cv2.imshow("ratio_img", ratio_img)
    # cv2.waitKey(0)