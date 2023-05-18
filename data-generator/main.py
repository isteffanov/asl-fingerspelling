import cv2 as cv
import time


def capture_photo():
    num_photos = 2

    capture = cv.VideoCapture(0)

    # Check if the camera is opened successfully
    if not capture.isOpened():
        print("Failed to open camera")
        return

    # Set the capture width and height
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    for i in range(num_photos):
        # we can also have an infinite loop
        # while True:
        ok, frame = capture.read()

        if not ok:
            print("Failed to take a photo, trying again in 1 second...")
            time.sleep(1)
            continue

        # show the pic
        # cv.imshow("Camera", frame)

        # save the pic
        file_name = f"data/photo_{int(time.time())}.jpg"
        cv.imwrite(file_name, frame)

        time.sleep(5)

        # quit?
        if cv.waitKey(1) == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()


capture_photo()
