import cv2
from time 	import time
import numpy as np


def preprocess(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (3,3), 0)
	edge = cv2.Canny(blur, 30, 40)
	return edge

def get_distance(point_a, point_b):
	x1, y1 = point_a
	x2, y2 = point_b
	return ((x1-x2)**2 + (y1-y2)**2)**0.5

def get_similatiry(distances):
	similatiry = 0
	n = len(distances)
	count = 0
	for i in range(n-1):
		i_dist = distances[i]
		for j in range(i+1, n):
			if i != j:
				j_dist = distances[j]
				ratio = min(i_dist, j_dist) / max(i_dist, j_dist)
				similatiry += ratio
				count += 1

	return similatiry/count




def detect_squares(img):
	squares = []
	contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		area = cv2.contourArea(cnt, True)
		if area > 100:
			perimeter = cv2.arcLength(cnt, True) * 0.1
			points = cv2.approxPolyDP(cnt, perimeter, True)
			if len(points) == 4:

				points = np.reshape(points, (4,2))
				p1,p2,p3,p4 = points

				distances = []
				distances.append(get_distance(p1,p2))
				distances.append(get_distance(p2,p3))
				distances.append(get_distance(p3,p4))
				distances.append(get_distance(p4,p1))

				similatiry = get_similatiry(distances)
				if similatiry > 0.95:
					squares.append(cnt)

	return squares



def run(mirror=False):
	cam = cv2.VideoCapture(0)
	
	old_frame_t = 0
	new_frame_t = 0
	while True:
		_, img = cam.read()
		if mirror:
			img = cv2.flip(img, 1)

		img_preprocessed = preprocess(img)

		squares = detect_squares(img_preprocessed)

		tmp = []
		for i in range(len(squares)):
			i_area = cv2.contourArea(squares[i])
			count = 0
			for j in range(len(squares)):
				if i != j:
					j_area = cv2.contourArea(squares[j])
					similarity_ratio = min(i_area, j_area) / max(i_area, j_area)
					if 0.75 < similarity_ratio <= 1:
						count += 1

			if count >= 4:
				tmp.append(squares[i])

		squares = tmp[:9]
		print(len(squares))
		centroids = []
		for cnt in squares:
			area = cv2.contourArea(cnt, True)
			peri = cv2.arcLength(cnt, True) * 0.1
			approx = cv2.approxPolyDP(cnt, peri,True)

			M = cv2.moments(cnt)

			if M['m00']:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				centroids.append([cx,cy])
				img = cv2.circle(img, (cx,cy), radius=4, color=(0, 0, 255), thickness=-1)

			cv2.drawContours(img, approx, -1, (0, 255, 0), 3)

		# centroids.sort(key=lambda x: (x[0],x[1]))
		
		# calculating fps
		new_frame_t = time()
		fps = 1/(new_frame_t-old_frame_t)
		fps = str(round(fps, 2))
		cv2.putText(img, fps, (0, 25), 0, 1, (0, 255, 0), 1, cv2.LINE_AA)
		old_frame_t = new_frame_t

		cv2.imshow('webcam', img)

		if cv2.waitKey(1) == 27:
			break

	cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	run(mirror=True)