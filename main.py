import cv2


def run(mirror=False):
	cam = cv2.VideoCapture(0)
	
	while True:
		_, img = cam.read()
		if mirror:
			img = cv2.flip(img, 1)

		cv2.imshow('webcam', img)

		if cv2.waitKey(1) == 27:
			break

	cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	run(mirror=True)