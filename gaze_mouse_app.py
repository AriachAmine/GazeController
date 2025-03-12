import cv2
import numpy as np
import pyautogui
import keyboard
from gaze_controller import GazeMouseController

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Get camera frame dimensions
    ret, test_frame = cap.read()
    if not ret:
        print("Failed to capture video from camera")
        return
    
    frame_height, frame_width = test_frame.shape[:2]
    
    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    
    # Create a named window that we can resize
    window_name = "Gaze Mouse Controller"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set the window to fullscreen
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Create GazeMouseController instance with more stable settings
    controller = GazeMouseController(
        screen_width=screen_width,
        screen_height=screen_height,
        smoothing_factor=0.4,  # Increased for smoother movement
        dwell_time=1.0,  # 1 second to trigger a click
        dwell_radius=30,  # 30 pixel radius for dwell area
    )

    # Application state variables
    running = True
    calibration_mode = False
    current_calibration_point = None
    last_gaze_point = None
    calibration_count = 0
    total_calibration_points = controller.calibration_points
    fullscreen_mode = True

    print("\n=== Gaze Mouse Controller ===")
    print("Press 'c' to start calibration")
    print("Press 'e' to enable/disable mouse control")
    print("Press 'f' to toggle fullscreen")
    print("Press 'ESC' to exit fullscreen")
    print("Press 'q' to quit")

    while running:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a more intuitive selfie-view
        frame = cv2.flip(frame, 1)
        
        # Create a larger canvas for display that matches screen dimensions
        display_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        
        # Calculate scaling to fit camera frame in the display window
        scale_factor = min(screen_width / frame_width, screen_height / frame_height)
        target_width = int(frame_width * scale_factor)
        target_height = int(frame_height * scale_factor)
        
        # Center the camera frame in the display
        x_offset = (screen_width - target_width) // 2
        y_offset = (screen_height - target_height) // 2
        
        # Resize the camera frame
        resized_frame = cv2.resize(frame, (target_width, target_height))
        
        # Place the resized frame in the center of the display
        display_frame[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized_frame

        # Process frame with gaze controller
        processed_frame, gaze_point = controller.process_frame(frame)
        
        # Process the display frame similarly
        if gaze_point is not None:
            # Scale the gaze point to the display frame
            display_gaze_x = int(gaze_point[0] * scale_factor) + x_offset
            display_gaze_y = int(gaze_point[1] * scale_factor) + y_offset
            
            # Draw gaze point on the display frame
            cv2.circle(display_frame, (display_gaze_x, display_gaze_y), 15, (0, 255, 0), -1)
            
            # Save the scaled gaze point for calibration
            last_gaze_point = (display_gaze_x, display_gaze_y)
        
        # Handle calibration mode
        if calibration_mode and current_calibration_point:
            # Draw current calibration point directly on the display frame
            cv2.circle(
                display_frame, current_calibration_point, 20, (0, 255, 255), -1
            )
            cv2.circle(
                display_frame, current_calibration_point, 7, (0, 0, 0), -1
            )

            # Display instruction and calibration status
            cv2.putText(
                display_frame,
                f"Look at the yellow point and press SPACE ({calibration_count+1}/{total_calibration_points})",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            
            # Display gaze status for debugging
            status_text = "Gaze detected" if gaze_point else "No gaze detected - face may not be visible"
            cv2.putText(
                display_frame,
                status_text,
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0) if gaze_point else (0, 0, 255),
                2,
            )

        # Add status text to display frame
        if controller.is_enabled:
            cv2.putText(display_frame, "Mouse Control: ON", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Mouse Control: OFF", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Show the display frame
        cv2.imshow(window_name, display_frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # ESC key exits fullscreen
        if key == 27:  # ESC key
            if fullscreen_mode:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                fullscreen_mode = False
        
        # Toggle fullscreen with 'f' key
        if keyboard.is_pressed("f"):
            fullscreen_mode = not fullscreen_mode
            if fullscreen_mode:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.waitKey(300)

        # Toggle mouse control with 'e' key
        if keyboard.is_pressed("e"):
            is_enabled = controller.toggle_enabled()
            print(f"Mouse control: {'ENABLED' if is_enabled else 'DISABLED'}")
            # Add small delay to prevent multiple toggles
            cv2.waitKey(300)

        # Start calibration with 'c' key
        if keyboard.is_pressed("c") and not calibration_mode:
            calibration_mode = True
            # Get calibration points designed for full screen
            current_calibration_point = controller.start_calibration()
            calibration_count = 0
            print("Calibration started. Look at the yellow point and press SPACE.")
            # Add small delay to prevent multiple toggles
            cv2.waitKey(300)

        # Capture calibration point with SPACE key
        if calibration_mode and keyboard.is_pressed("space"):
            point_to_use = last_gaze_point
            
            if point_to_use:
                next_point = controller.add_calibration_point(point_to_use)
                calibration_count += 1
                
                if next_point:
                    current_calibration_point = next_point
                    print(f"Point {calibration_count} recorded. Look at the next yellow point and press SPACE.")
                else:
                    calibration_mode = False
                    print("Calibration completed!")
            else:
                print("Cannot calibrate - no gaze detected. Make sure your face is visible.")
                
            # Add small delay to prevent multiple captures
            cv2.waitKey(300)

        # Skip current calibration point with 's' key
        if calibration_mode and keyboard.is_pressed("s"):
            if calibration_count < total_calibration_points - 1:
                calibration_count += 1
                current_calibration_point = controller.get_calibration_positions()[calibration_count]
                print(f"Skipped to point {calibration_count+1}")
            else:
                calibration_mode = False
                print("Calibration canceled - not enough points collected")
            cv2.waitKey(300)

        # Quit with 'q' key
        if keyboard.is_pressed("q"):
            running = False

    # Clean up
    controller.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
