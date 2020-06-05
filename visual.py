import numpy as np
import cv2

def draw_points(p, w, labels, nlabels):
    img = np.zeros((w, w, 3), np.uint8)
    assert len(p.shape) == 2 and p.shape[1] == 2, 'draw_points() expects 2-dim data'
    
    #rescale points
    xmin, xmax, ymin, ymax = min(min(p[:,0]), -1.), max(max(p[:,0]), 1.), min(min(p[:,1]), -1.), max(max(p[:,1]), 1.)
    limits = np.array([xmin, ymin])
    s = max(xmax - xmin, ymax - ymin)
    scale = np.array([w / s, w / s])
    p = np.int32((p[:, :2] - limits) * scale)
    
    #generate label colormap and labeled set
    if nlabels > 0:
        cmap = {}
        for i in range(nlabels+1):
            np.random.seed(i)
            color  = np.random.randint(255, size=3)
            cmap[i] = (int(color[0]), int(color[1]), int(color[2]))
        labeled = np.append(p, labels[:,None], axis=1)
        for x, y, label in labeled:
            cv2.circle(img, (x, y), 2, cmap[label], -1, cv2.LINE_AA, shift=0)
    else:
        for x, y in p:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA, shift=0)

    #draw unit square for reference
    xt, yt = np.int32((np.array([-.98, -.98]) - limits) * scale)
    xb, yb  = np.int32((np.array([.98, .98]) - limits) * scale)
    cv2.rectangle(img, (xt, yt), (xb, yb), (255, 255, 255), 3, cv2.LINE_AA)
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
    assert len(zs.shape) == 2 and zs.shape[1] == 2, 'covered_area() expects 2-dim data'
    r = radius
    m = np.zeros((resolution, resolution))
    for z in zs:
        zr = (resolution * (z + 1) / 2).astype(int)
        x, y = zr
        if x > 0 and y > 0:
            m[max((x-r, 0)):min((x+r, resolution)), max((y-r, 0)):min((y+r, resolution))] = 1
    return np.mean(m.astype(np.float32))
