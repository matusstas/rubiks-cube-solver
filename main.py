import cv2
from time 	import time
import numpy as np


def preprocess(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (3,3), 0)
	edge = cv2.Canny(blur, 30, 40)
	return edge


def detect_squares(img):
	squares = []
	contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		area = cv2.contourArea(cnt, True)
		if area > 0:
			perimeter = cv2.arcLength(cnt, True) * 0.1
			points = cv2.approxPolyDP(cnt, perimeter, True)
			if len(points) == 4:
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

		# calculating fps
		new_frame_t = time()
		fps = 1/(new_frame_t-old_frame_t)
		fps = str(round(fps, 2))
		cv2.putText(img, fps, (0, 25), 0, 1, (0, 255, 0), 1, cv2.LINE_AA)
		old_frame_t = new_frame_t

		img_preprocessed = preprocess(img)

		squares = detect_squares(img_preprocessed)

		squares.sort(key=lambda x: cv2.contourArea(x), reverse=True)
		for cnt in squares[:9]:
			peri = cv2.arcLength(cnt, True) * 0.1
			approx = cv2.approxPolyDP(cnt, peri,True)

			M = cv2.moments(cnt)

			if M['m00']:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				img = cv2.circle(img, (cx,cy), radius=4, color=(0, 0, 255), thickness=-1)

			cv2.drawContours(img, approx, -1, (0, 255, 0), 3)

		cv2.imshow('webcam', img)

		if cv2.waitKey(1) == 27:
			break

	cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	run(mirror=True)