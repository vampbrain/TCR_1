import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(0)

dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2, k3])
marker_size = 0.1

while True:
    ret, frame = cap.read()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary, parameters=parameters)

    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            cv2.putText(frame, str(ids[i]), tuple(corners[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size*0.5)

        frame = aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()