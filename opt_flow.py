#!/usr/bin/env python

'''

Based on opencv dense optical flow example from
https://github.com/opencv/opencv/blob/10fb88d02791b33d83a3756c62e21aa1c5a1e68d/samples/python/opt_flow.py#L3


example to show optical flow
USAGE: opt_flow.py
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang*(180/np.pi/2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res


def avg_flow(flow):
    return (np.mean(flow[:, :, 0]), np.mean(flow[:, :, 1]))


def crop_img(img, flow_x, flow_y, width, height):
        ymin = int(0 if -flow_y < 0 else -flow_y)
        ymax = int(height if height - flow_y > height else height - flow_y)
        xmin = int(0 if -flow_x < 0 else -flow_x)
        xmax = int(width if width - flow_x > width else width - flow_x)

        return img[ymin:ymax, xmin:xmax]


SQUARE_SIZE = 159


if __name__ == '__main__':
    print(__doc__)

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_BUFFERSIZE, 1)
    cols = width = np.int32(cam.get(3))
    rows = height = np.int32(cam.get(4))
    screen_proportion = abs(cols / rows)
    print('resolucao:', cols, rows)
    x_center = cols // 2
    y_center = rows // 2

    ret, prev = cam.read()
    full_img = prev
    prev = prev[x_center-SQUARE_SIZE:x_center+SQUARE_SIZE,
                y_center-SQUARE_SIZE:y_center+SQUARE_SIZE]
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()

    flow = None

    while True:
        ret, img = cam.read()
        gray_full = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = gray_full[x_center-SQUARE_SIZE:x_center+SQUARE_SIZE,
                         y_center-SQUARE_SIZE:y_center+SQUARE_SIZE]
        if flow is None:
            flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5,
                                               3, 15, 3, 5, 1.2, 0)
        else:
            flow += cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5,
                                                3, 15, 3, 5, 1.2, 0)

        print(flow.shape)
        prevgray = gray

        flow_x, flow_y = avg_flow(flow)
        print(flow_x, flow_y)

        if abs(flow_x / flow_y) > screen_proportion:
            flow_y = screen_proportion / flow_x
        else:
            flow_x = screen_proportion * flow_y

        M = np.array([[1, 0, -flow_x], [0, 1, -flow_y]],
                     dtype=np.float32)

        img_shifted = cv.cvtColor(gray_full, cv.COLOR_GRAY2RGB)
        gray_rectangle = cv.rectangle(
            img_shifted,
            (x_center-SQUARE_SIZE, y_center-SQUARE_SIZE),
            (x_center+SQUARE_SIZE, y_center+SQUARE_SIZE),
            (0, 0, 255),
            3
        )

        gray_rectangle = cv.warpAffine(gray_rectangle, M, (cols, rows))

        crop = crop_img(gray_rectangle, flow_x, flow_y, width, height)
        crop = cv.resize(crop, (width, height))

        cv.imshow('corrigida', cv.flip(crop, 1))
        if show_hsv:
            cv.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv.imshow('glitch', cur_glitch)

        ch = cv.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
                print('glitch is', ['off', 'on'][show_glitch])
                cv.destroyAllWindows()
