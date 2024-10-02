import cv2


def camera():
    cap = cv2.VideoCapture(0)


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
       
        cv2.imshow('Capture image', frame)


        if cv2.waitKey(1) == ord('s'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28))
            cv2.imwrite('captured_image.png', resized)
            print('Image captured and saved')
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera()
