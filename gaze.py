import cv2
import numpy as np
from helpers import relative, relativeT


def calculate_thresholds(
    screen_width,
    screen_height,
    gaze_deviation_percentage_x=0.1,
    gaze_deviation_percentage_y=0.1,
):
    """
    Calculates the threshold values for gaze deviation based on the screen dimensions.

    Args:
        screen_width (int): The width of the screen in pixels.
        screen_height (int): The height of the screen in pixels.
        gaze_deviation_percentage_x (float): The percentage of the screen width that represents the maximum allowed
                                              gaze deviation in the x-axis.  A value of 0.1 means 10% of the screen width.
        gaze_deviation_percentage_y (float): The percentage of the screen height that represents the maximum allowed
                                              gaze deviation in the y-axis.  A value of 0.1 means 10% of the screen height.

    Returns:
        tuple: A tuple containing the calculated threshold_x and threshold_y values.
    """

    threshold_x = int(screen_width * gaze_deviation_percentage_x)
    threshold_y = int(screen_height * gaze_deviation_percentage_y)

    return threshold_x, threshold_y


def gaze(frame, points, eye_movement_weight=0.7):
    """
    The gaze function gets an image and face landmarks from mediapipe framework.
    The function draws the gaze direction into the frame.
    Returns the (x,y) coordinates of the gaze direction.  Returns (-1, -1) on error.
    
    Args:
        frame: The video frame
        points: Facial landmarks from MediaPipe
        eye_movement_weight: Weight given to eye movement vs head position (0-1)
    """
    try:
        # Track pupils and eye corners for enhanced eye movement detection
        left_pupil = relative(points.landmark[468], frame.shape)
        right_pupil = relative(points.landmark[473], frame.shape)
        
        # Eye corners
        left_eye_inner = relative(points.landmark[133], frame.shape)
        left_eye_outer = relative(points.landmark[33], frame.shape)
        right_eye_inner = relative(points.landmark[362], frame.shape)
        right_eye_outer = relative(points.landmark[263], frame.shape)
        
        # Calculate eye centers
        left_eye_center = ((left_eye_inner[0] + left_eye_outer[0])//2,
                           (left_eye_inner[1] + left_eye_outer[1])//2)
        right_eye_center = ((right_eye_inner[0] + right_eye_outer[0])//2,
                           (right_eye_inner[1] + right_eye_outer[1])//2)
        
        # Calculate pupil offset from center of eye (this detects eye movement)
        left_pupil_offset_x = left_pupil[0] - left_eye_center[0]
        left_pupil_offset_y = left_pupil[1] - left_eye_center[1]
        
        right_pupil_offset_x = right_pupil[0] - right_eye_center[0]
        right_pupil_offset_y = right_pupil[1] - right_eye_center[1]
        
        # Average pupil offset
        avg_pupil_offset_x = (left_pupil_offset_x + right_pupil_offset_x) / 2
        avg_pupil_offset_y = (left_pupil_offset_y + right_pupil_offset_y) / 2
        
        # Scale up pupil offset to make it more impactful
        pupil_scale_factor = 5.0
        scaled_offset_x = avg_pupil_offset_x * pupil_scale_factor
        scaled_offset_y = avg_pupil_offset_y * pupil_scale_factor

        """
        2D image points for head pose estimation
        """
        image_points = np.array(
            [
                relative(points.landmark[4], frame.shape),  # Nose tip
                relative(points.landmark[152], frame.shape),  # Chin
                relative(points.landmark[263], frame.shape),  # Left eye left corner
                relative(points.landmark[33], frame.shape),  # Right eye right corner
                relative(points.landmark[287], frame.shape),  # Left Mouth corner
                relative(points.landmark[57], frame.shape),  # Right mouth corner
            ],
            dtype="double",
        )

        """
        2D image points at (x,y,0) format
        """
        image_points1 = np.array(
            [
                relativeT(points.landmark[4], frame.shape),  # Nose tip
                relativeT(points.landmark[152], frame.shape),  # Chin
                relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
                relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
                relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
                relativeT(points.landmark[57], frame.shape),  # Right mouth corner
            ],
            dtype="double",
        )

        # 3D model points.
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0, -63.6, -12.5),  # Chin
                (-43.3, 32.7, -26),  # Left eye, left corner
                (43.3, 32.7, -26),  # Right eye, right corner
                (-28.9, -28.9, -24.1),  # Left Mouth corner
                (28.9, -28.9, -24.1),  # Right mouth corner
            ]
        )

        """
        3D model eye points
        """
        Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
        Eye_ball_center_left = np.array(
            [[29.05], [32.7], [-39.5]]
        )  # the center of the left eyeball as a vector.

        """
        camera matrix estimation
        """
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # Transformation between image point to world point
        _, transformation, _ = cv2.estimateAffine3D(
            image_points1, model_points
        )  # image to world transformation

        if transformation is not None:  # if estimateAffine3D succeeded
            # project pupil image point into 3d world point
            pupil_world_cord = (
                transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
            )

            # 3D gaze point (10 is arbitrary value denoting gaze distance)
            S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10

            # Project a 3D gaze direction onto the image plane.
            (eye_pupil2D, _) = cv2.projectPoints(
                (int(S[0]), int(S[1]), int(S[2])),
                rotation_vector,
                translation_vector,
                camera_matrix,
                dist_coeffs,
            )
            # project 3D head pose into the image plane
            (head_pose, _) = cv2.projectPoints(
                (int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                rotation_vector,
                translation_vector,
                camera_matrix,
                dist_coeffs,
            )
            
            # Original gaze direction calculation
            basic_gaze_direction = (
                left_pupil
                + (eye_pupil2D[0][0] - left_pupil)
                - (head_pose[0][0] - left_pupil)
            )
            
            # Enhance with pupil tracking (blend the pupil offset with the PnP-based gaze)
            enhanced_gaze_x = basic_gaze_direction[0] + scaled_offset_x
            enhanced_gaze_y = basic_gaze_direction[1] + scaled_offset_y
            
            # Blend between standard gaze detection and enhanced pupil tracking
            # Higher eye_movement_weight gives more importance to pupil movement
            final_gaze_x = int((1 - eye_movement_weight) * basic_gaze_direction[0] + 
                             eye_movement_weight * enhanced_gaze_x)
            final_gaze_y = int((1 - eye_movement_weight) * basic_gaze_direction[1] + 
                             eye_movement_weight * enhanced_gaze_y)

            # Draw gaze line into screen
            p1 = (int(left_pupil[0]), int(left_pupil[1]))
            p2 = (final_gaze_x, final_gaze_y)
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
            
            # Also draw pupil offset visualization (for debugging)
            cv2.circle(frame, left_eye_center, 3, (255, 0, 0), -1)  # Blue - eye center
            cv2.circle(frame, (int(left_eye_center[0] + left_pupil_offset_x*3), 
                              int(left_eye_center[1] + left_pupil_offset_y*3)), 
                      3, (0, 255, 255), -1)  # Yellow - scaled pupil position

            return final_gaze_x, final_gaze_y  # Return as integers
        else:
            print("transformation is None")
            return -1, -1  # Error signal

    except Exception as e:
        print(f"Error in gaze estimation: {e}")
        return -1, -1  # Indicate an error
