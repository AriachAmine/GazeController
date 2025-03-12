# Gaze Controller - Control Your Mouse With Your Eyes

Gaze Controller is a Python-based application that allows you to control your computer's mouse cursor using only your eye movements. This technology leverages computer vision and eye tracking to create an accessible hands-free mouse control system.

## Features

- **Eye Tracking**: Control your mouse cursor by simply looking at different parts of your screen
- **Dwell Clicking**: Hover your gaze over an element for a set time to perform a click
- **Calibration System**: Personalized calibration ensures accurate cursor positioning
- **Sensitivity Controls**: Adjust how responsive the cursor is to your eye movements
- **Smoothing Algorithms**: Implements multiple smoothing techniques for stable cursor movement
- **Toggle Controls**: Easily enable/disable the gaze control when needed
- **Fullscreen Mode**: Option to use the application in fullscreen for better visibility

## Requirements

- Python 3.7+
- Webcam
- The following Python packages:
  - OpenCV (cv2)
  - NumPy
  - MediaPipe
  - PyAutoGUI
  - keyboard

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/AriachAmine/GazeController.git
   cd GazeController
   ```

2. Install the required packages:
   ```
   pip install opencv-python numpy mediapipe pyautogui keyboard
   ```

## Usage

1. Run the application:
   ```
   python gaze_mouse_app.py
   ```

2. Position yourself in front of your webcam so that your face is clearly visible.

3. Use the keyboard shortcuts below to control the application.

## Controls

| Key | Function |
|-----|----------|
| `c` | Start calibration process |
| `e` | Enable/disable mouse control |
| `f` | Toggle fullscreen mode |
| `ESC` | Exit fullscreen mode |
| `SPACE` | During calibration: confirm looking at the current calibration point |
| `s` | During calibration: skip the current calibration point |
| `q` | Quit application |
| `+` | Increase cursor sensitivity |
| `-` | Decrease cursor sensitivity |

## Calibration

For best results, calibrate the system before use:

1. Press `c` to start the calibration process
2. Look at each yellow calibration point that appears on the screen
3. Press `SPACE` while looking at each point
4. After calibrating all points, the system will be ready to use

If you find that cursor movement requires too much head movement:
1. Try increasing the sensitivity using the `+` key
2. Make sure to calibrate with subtle eye movements rather than large head movements
3. Position your face at a comfortable distance from the camera

## How It Works

The application uses:

1. **MediaPipe Face Mesh** to detect facial landmarks
2. **Custom gaze estimation** algorithms to determine where you're looking
3. **Pupil tracking** to enhance detection of subtle eye movements
4. **Coordinate mapping** to translate eye movements to screen positions
5. **Smoothing algorithms** to reduce jitter and provide stable cursor control

## Advanced Configuration

You can adjust the following parameters in `gaze_controller.py`:

- `smoothing_factor`: Controls how smooth the cursor movement is (0-1)
- `dwell_time`: Time in seconds required to trigger a click
- `dwell_radius`: Pixel radius for dwell detection area
- `calibration_points`: Number of points used for calibration (4 or 9)
- `sensitivity`: How responsive the cursor is to eye movements (1.0-5.0)

## Troubleshooting

- **Limited cursor movement**: Increase sensitivity using the `+` key
- **No face detected**: Make sure your face is clearly visible to the webcam and there is adequate lighting
- **Cursor movement is erratic**: Recalibrate the system and ensure stable lighting conditions
- **Application is slow**: Reduce the resolution of your webcam or close other resource-intensive applications

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for their face mesh detection framework
- The computer vision community for eye tracking research