import cv2
import math
import numpy as np
from scipy.interpolate import splprep, splev


class Hand:

    def __init__(self, binary, masked, raw, frame, original):
        self.masked = masked
        self.binary = binary
        self._raw = raw
        self.frame = frame
        self.contours = []
        self.original = original
        self.outline = self.draw_outline()
        self.fingertips, self.count = self.extract_fingertips()

    def draw_outline(self, min_area=10000, color=(0, 255, 0), thickness=2):
        contours, _ = cv2.findContours(
            self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        palm_area = 0
        flag = None
        cnt = None
        for (i, c) in enumerate(contours):
            area = cv2.contourArea(c)
            if area > palm_area:
                palm_area = area
                flag = i
        if flag is not None and palm_area > min_area:
            cnt = contours[flag]
            self.contours = cnt
            cpy = self.original.copy()
            cv2.drawContours(cpy, [cnt], 0, color, thickness)
            return cpy
        else:
            return self.original

    def extract_fingertips(self, filter_value=50):
        count = 0
        cnt = self.contours
        if len(cnt) == 0:
            return (cnt, count)
        points = []
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            end = tuple(cnt[e][0])
            points.append(end)

            start = tuple(cnt[s][0])
            far = tuple(cnt[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
            if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                count += 1

        filtered = self.filter_points(points, filter_value)

        filtered.sort(key=lambda point: point[1])
        return ([pt for idx, pt in zip(range(5), filtered)], count)

    def filter_points(self, points, filter_value):
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if points[i] and points[j] and self.dist(points[i], points[j]) < filter_value:
                    points[j] = None
        filtered = []
        for point in points:
            if point is not None:
                filtered.append(point)
        return filtered

    def get_center_of_mass(self):
        if len(self.contours) == 0:
            return None
        M = cv2.moments(self.contours)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    def dist(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (b[1] - a[1])**2)
