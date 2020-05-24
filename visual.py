import numpy as np
import cv2


def draw_points(p, w):
    img = np.zeros((w, w, 3), np.uint8)
    p = np.int32(w / 2 + p[:, :2] * w / 2)
    # print(p)
    for x, y in p:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA, shift=0)
    return img

def draw_edges(p1, p2, w, radius, edges=True):
    img = np.zeros((w, w, 3), np.uint8)

    p1 = p1 / radius
    p2 = p2 / radius

    p1 = np.int32(w / 2 + p1[:, :2] * w / 2)
    p2 = np.int32(w / 2 + p2[:, :2] * w / 2)
    if edges:
        for (x1, y1), (x2, y2) in zip(p1, p2):
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    for (x2, y2) in p2:
        cv2.circle(img, (x2, y2), 5, (0, 0, 255), -1, cv2.LINE_AA)
    for (x1, y1) in p1:
        cv2.circle(img, (x1, y1), 3, (255, 0, 0), -1, cv2.LINE_AA)
    return img

def covered_area(zs, resolution=400, radius=10):
    r = radius
    m = np.zeros((resolution, resolution))
    for z in zs:
        zr = (resolution * (z + 1) / 2).astype(int)
        x, y = zr
        m[max((x-r, 0)):min((x+r, resolution)), max((y-r, 0)):min((y+r, resolution))] += 1
    return np.mean((m > 0).astype(np.float32))
