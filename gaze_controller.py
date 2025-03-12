import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
from gaze import gaze
from collections import deque

class GazeMouseController:
    def __init__(
        self,
        screen_width=None,
        screen_height=None,
        smoothing_factor=0.3,
        dwell_time=1.0,
        dwell_radius=50,
        calibration_points=4,
    ):
        """
        Initialize the gaze-controlled mouse system.
        
        Args:
            screen_width (int): Width of the screen. Defaults to pyautogui.size()[0].
            screen_height (int): Height of the screen. Defaults to pyautogui.size()[1].
            smoothing_factor (float): Factor for cursor movement smoothing (0-1).
            dwell_time (float): Time in seconds to dwell before clicking.
            dwell_radius (int): Radius in pixels for dwell detection.
            calibration_points (int): Number of calibration points (4 or 9).
        """
        # Initialize screen dimensions
        screen_size = pyautogui.size()
        self.screen_width = screen_width or screen_size[0]
        self.screen_height = screen_height or screen_size[1]
        
        # Mouse control parameters
        self.smoothing_factor = smoothing_factor
        self.dwell_time = dwell_time
        self.dwell_radius = dwell_radius
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # Tracking variables
        self.prev_cursor_pos = (self.screen_width // 2, self.screen_height // 2)
        self.cursor_pos = self.prev_cursor_pos
        self.dwell_start_time = None
        self.dwell_start_pos = None
        self.is_enabled = False
        
        # Calibration variables
        self.calibration_points = calibration_points
        self.calibration_data = []
        self.mapping_matrix = None
        self.is_calibrated = False
        self.calibration_point_index = 0
        
        # Disable pyautogui fail-safe
        pyautogui.FAILSAFE = False
        
        # Stabilization enhancement: Moving average filter
        self.gaze_history = deque(maxlen=10)  # Store last 10 gaze positions
        self.jitter_threshold = 5  # Pixels - ignore movements smaller than this
        self.last_stable_gaze = None
        
        # Exponential smoothing parameters
        self.alpha = 0.3  # Smoothing factor for exponential smoothing
        self.exp_smoothed_x = None
        self.exp_smoothed_y = None
    
    def get_calibration_positions(self):
        """Generate positions for calibration points."""
        w, h = self.screen_width, self.screen_height
        padding = min(w, h) // 10
        
        if self.calibration_points == 4:
            return [
                (padding, padding),                # Top-left
                (w - padding, padding),            # Top-right
                (padding, h - padding),            # Bottom-left
                (w - padding, h - padding),        # Bottom-right
            ]
        else:  # 9 points
            return [
                (padding, padding),                # Top-left
                (w // 2, padding),                 # Top-center
                (w - padding, padding),            # Top-right
                (padding, h // 2),                 # Middle-left
                (w // 2, h // 2),                 # Center
                (w - padding, h // 2),            # Middle-right
                (padding, h - padding),            # Bottom-left
                (w // 2, h - padding),            # Bottom-center
                (w - padding, h - padding),        # Bottom-right
            ]
    
    def start_calibration(self):
        """Start the calibration process."""
        self.calibration_data = []
        self.calibration_point_index = 0
        self.is_calibrated = False
        return self.get_calibration_positions()[0]
    
    def add_calibration_point(self, gaze_point):
        """Add a calibration point with corresponding gaze data."""
        if self.calibration_point_index < self.calibration_points:
            actual_point = self.get_calibration_positions()[self.calibration_point_index]
            self.calibration_data.append((gaze_point, actual_point))
            self.calibration_point_index += 1
            
            # If we've collected all points, compute the mapping
            if self.calibration_point_index >= self.calibration_points:
                self._compute_mapping()
                self.is_calibrated = True
                return None
            
            # Return the next calibration point
            return self.get_calibration_positions()[self.calibration_point_index]
        
        return None
    
    def _compute_mapping(self):
        """Compute the mapping matrix from gaze coordinates to screen coordinates."""
        if len(self.calibration_data) < 3:
            return False
            
        # Extract points into numpy arrays
        gaze_points = np.array([p[0] for p in self.calibration_data])
        screen_points = np.array([p[1] for p in self.calibration_data])
        
        # Add a column of ones to gaze_points for affine transformation
        gaze_points_homogeneous = np.hstack((gaze_points, np.ones((len(gaze_points), 1))))
        
        # Solve for the transformation matrix (least squares)
        self.mapping_matrix, _, _, _ = np.linalg.lstsq(gaze_points_homogeneous, screen_points, rcond=None)
        return True
    
    def map_gaze_to_screen(self, gaze_x, gaze_y):
        """Map gaze coordinates to screen coordinates using the calibration mapping."""
        if self.is_calibrated and self.mapping_matrix is not None:
            gaze_point_homogeneous = np.array([gaze_x, gaze_y, 1])
            screen_x, screen_y = np.dot(gaze_point_homogeneous, self.mapping_matrix)
            return max(0, min(int(screen_x), self.screen_width)), max(0, min(int(screen_y), self.screen_height))
        else:
            # Simple mapping if no calibration
            scale_x = self.screen_width / 640
            scale_y = self.screen_height / 480
            return int(gaze_x * scale_x), int(gaze_y * scale_y)
    
    def process_frame(self, frame):
        """
        Process a frame to track gaze and control the mouse cursor.
        
        Args:
            frame: The video frame to process
            
        Returns:
            tuple: (processed_frame, gaze_position)
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not results.multi_face_landmarks:
            return frame, None
            
        landmarks = results.multi_face_landmarks[0]
        
        # Use the existing gaze function to get gaze direction
        gaze_x, gaze_y = gaze(frame, landmarks)
        
        if gaze_x < 0 or gaze_y < 0:  # Error case in gaze detection
            return frame, None
            
        # Stabilization enhancement: Add to moving average
        self.gaze_history.append((gaze_x, gaze_y))
        
        # Calculate moving average for more stable gaze point
        if len(self.gaze_history) > 3:  # Only after we have enough samples
            avg_gaze_x = sum(point[0] for point in self.gaze_history) / len(self.gaze_history)
            avg_gaze_y = sum(point[1] for point in self.gaze_history) / len(self.gaze_history)
            
            # Further smooth with exponential smoothing
            if self.exp_smoothed_x is None:
                self.exp_smoothed_x = avg_gaze_x
                self.exp_smoothed_y = avg_gaze_y
            else:
                self.exp_smoothed_x = self.alpha * avg_gaze_x + (1 - self.alpha) * self.exp_smoothed_x
                self.exp_smoothed_y = self.alpha * avg_gaze_y + (1 - self.alpha) * self.exp_smoothed_y
            
            # Use the smoothed values
            smooth_gaze_x = int(self.exp_smoothed_x)
            smooth_gaze_y = int(self.exp_smoothed_y)
        else:
            # Not enough samples yet, use the raw values
            smooth_gaze_x = gaze_x
            smooth_gaze_y = gaze_y
        
        # Stabilization enhancement: Ignore jitter
        if self.last_stable_gaze is not None:
            dx = abs(smooth_gaze_x - self.last_stable_gaze[0])
            dy = abs(smooth_gaze_y - self.last_stable_gaze[1])
            
            # If movement is too small (likely jitter), keep the previous stable position
            if dx < self.jitter_threshold and dy < self.jitter_threshold:
                smooth_gaze_x, smooth_gaze_y = self.last_stable_gaze
            else:
                self.last_stable_gaze = (smooth_gaze_x, smooth_gaze_y)
        else:
            self.last_stable_gaze = (smooth_gaze_x, smooth_gaze_y)
            
        # Draw gaze point on the frame for reference
        cv2.circle(frame, (smooth_gaze_x, smooth_gaze_y), 10, (0, 255, 0), -1)
        
        if self.is_enabled:
            # Map gaze to screen coordinates
            screen_x, screen_y = self.map_gaze_to_screen(smooth_gaze_x, smooth_gaze_y)
            
            # Apply additional smoothing to cursor movement - more weight to previous position
            smoothed_x = int(self.prev_cursor_pos[0] * (1 - self.smoothing_factor) + screen_x * self.smoothing_factor)
            smoothed_y = int(self.prev_cursor_pos[1] * (1 - self.smoothing_factor) + screen_y * self.smoothing_factor)
            
            # Move the cursor to the smoothed position
            self.cursor_pos = (smoothed_x, smoothed_y)
            pyautogui.moveTo(smoothed_x, smoothed_y)
            
            # Dwell click detection
            if self.dwell_start_pos is None:
                self.dwell_start_pos = self.cursor_pos
                self.dwell_start_time = time.time()
            else:
                # Calculate distance from dwell start position
                dx = self.cursor_pos[0] - self.dwell_start_pos[0]
                dy = self.cursor_pos[1] - self.dwell_start_pos[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # If cursor moved outside the dwell radius, reset dwell detection
                if distance > self.dwell_radius:
                    self.dwell_start_pos = self.cursor_pos
                    self.dwell_start_time = time.time()
                elif time.time() - self.dwell_start_time > self.dwell_time:
                    # Click if dwell time is reached
                    pyautogui.click(self.cursor_pos[0], self.cursor_pos[1])
                    # Reset after click
                    self.dwell_start_pos = None
                    self.dwell_start_time = None
                    
                    # Draw a circle to indicate click
                    cv2.circle(frame, (smooth_gaze_x, smooth_gaze_y), 30, (0, 0, 255), 3)
            
            # Save current position for next frame's smoothing
            self.prev_cursor_pos = self.cursor_pos
            
            # Draw visualization
            cv2.putText(frame, "Mouse Control: ON", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Mouse Control: OFF", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return frame, (smooth_gaze_x, smooth_gaze_y)
        
    def toggle_enabled(self):
        """Toggle mouse control on/off."""
        self.is_enabled = not self.is_enabled
        if not self.is_enabled:
            self.dwell_start_pos = None
            self.dwell_start_time = None
        return self.is_enabled
        
    def release(self):
        """Release resources."""
        self.face_mesh.close()
