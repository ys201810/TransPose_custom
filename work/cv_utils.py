# coding=utf-8
import cv2


def draw_circle(img, coordinate, radius=5, color=(255, 255, 255), thickness=1):
    return cv2.circle(img, coordinate, radius=radius, color=color,
                      thickness=thickness, lineType=cv2.LINE_4, shift=0)

