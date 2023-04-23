import cv2
import numpy as np
# Define the size of the checkerboard
pattern_size = (9, 6)

# Define the object points
object_points = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Define the arrays to store object points and image points from all images
obj_points = [] # 3D points in real world space
img_points = [] # 2D points in image plane

# Capture the video frame
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Find the chessboard corners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # Refine the corner locations
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Draw the corners on the image
        cv2.drawChessboardCorners(frame, pattern_size, corners, ret)

        # Add the object points and image points
        obj_points.append(object_points)
        img_points.append(corners)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Print the camera matrix and distortion coefficients
print('Camera matrix:')
print(camera_matrix)
print('Distortion coefficients:')
print(dist_coeffs)
# Define the Aruco dictionary
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Initialize the detector
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

# Define the camera matrix and distortion coefficients
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# Capture the video frame
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect the marker
    marker_corners, marker_ids, _ = detector.detectMarkers(frame)

    if marker_ids is not None:
        # Estimate the pose
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners, 0.05, camera_matrix, dist_coeffs)
        rvec, tvec = rvecs[0], tvecs[0]

        # Display the ID
        cv2.putText(frame, str(marker_ids[0]), tuple(marker_corners[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw the axes
        axis_len = 0.1
        axis_points = np.float32([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]]).reshape(-1, 3)
        axis_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
        origin = tuple(map(int, marker_corners[0][0]))
        x, y = axis_points[1][0], axis_points[2][0]
        cv2.line(frame, origin, tuple(map(int, x)), (255, 0, 0), 3)
        cv2.line(frame, origin, tuple(map(int, y)), (0, 255, 0), 3)
        cv2.line(frame, origin, tuple(map(int, axis_points[3][0])), (0, 0, 255), 3)

        # Highlight the marker
        cv2.rectangle(frame, tuple(map(int, marker_corners[0][0])), tuple(map(int, marker_corners[0][2])), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
