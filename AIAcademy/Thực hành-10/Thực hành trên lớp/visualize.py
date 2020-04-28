import numpy as np
import cv2
import random


def visualize(graph_name, label):
    img_size = 600
    img = np.zeros((img_size, img_size, 3), np.uint8) + 255

    file_path = graph_name + '.vis'
    with open(file_path, 'r') as f:
        n_cluster = int(next(f).split()[0])
        cl = [int(x) for x in next(f).split()]
    n = np.sum(cl)
    m = int(np.sqrt(n_cluster-1)) + 1
    cluster_size = img_size / m
    margin = cluster_size / 10

    # Generate random n point (xx, yy)
    xx = [0] * n
    yy = [0] * n
    cnt = 0
    for i in range(n_cluster):
        xtl = int(i / m) * cluster_size
        ytl = int(i % m) * cluster_size
        for j in range(cl[i]):
            xx[cnt + j] = int(xtl + random.randint(margin, cluster_size - margin))
            yy[cnt + j] = int(ytl + random.randint(margin, cluster_size - margin))
        cnt += cl[i]

    # Draw edges
    file_path = graph_name + '.txt'
    with open(file_path, 'r') as f:
        n_, m_, k_ = [int(x) for x in next(f).split()]
        for line in f:
            u, v = [int(x) for x in line.split()]
            cv2.line(img, (xx[u], yy[u]), (xx[v], yy[v]), (0, 0, 0), 1)

    color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(30)]

    # Draw vertices with label
    for i in range(n):
        # cv2.circle(img, (xx[i], yy[i]), 11, (label[i]*255/n_cluster, 100, (n_cluster-label[i])*255/n_cluster), -1)
        cv2.circle(img, (xx[i], yy[i]), 11, color[label[i]], -1)
        cv2.circle(img, (xx[i], yy[i]), 12, (0, 0, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(i), (xx[i]-6, yy[i]+2), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(graph_name+'.jpg', img)
    return img

