
# Traffic Monitoring System with Automatic License Plate Recognition

## **Project Overview**
This project is a traffic monitoring system that uses computer vision to detect vehicle movement at a traffic light, determine if a vehicle crossed during a red light, and extract the license plate for issuing a fine if necessary. The system utilizes video streams and OpenCV to track vehicles and performs OCR (Optical Character Recognition) using Tesseract to read license plates.

---

## **Project Structure**

```
├── calibration.py       # Camera calibration functions
├── main.py              # Main execution script for traffic light monitoring
├── matric.py            # License plate detection and OCR functions
├── tracking.py          # Vehicle tracking logic
├── test.py              # Utility script for streaming and recording video
├── tracking_record.py   # Script to record the object tracking video output
├── grabar_video.py      # Script to record the annotated tracking video in .avi format
└── README.md            # Project documentation
```

---

## **Features**

- **Camera Calibration:** Uses chessboard patterns to calibrate the camera, compensating for lens distortions.
- **Vehicle Tracking:** Detects moving vehicles using background subtraction and tracks them with a Kalman filter.
- **Traffic Light Monitoring:** Determines whether vehicles crossed the line when the light was red.
- **License Plate Recognition:** Extracts the license plate using contour detection and performs OCR using Tesseract.
- **Video Recording and Playback:** Supports recording videos from the camera and playing them for analysis.
- **Object Tracking Recording:** Saves a video with annotated object tracking results for further analysis.

---

## **Dependencies**

Ensure the following Python packages are installed:

```bash
pip install numpy opencv-python imageio matplotlib imutils pytesseract scikit-image
```

Additionally:
- **Tesseract-OCR**: Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and configure the path in your script.

---

## **How to Run**

### 1. **Camera Calibration**
Before running the main script, calibrate the camera by using chessboard images.
- `calibration.py` handles this step:
  ```python
  from calibration import calibrar
  rms, intrinsics, extrinsics, dist_coeffs = calibrar()
  ```

### 2. **Main Monitoring Script**
To run the main traffic monitoring system:
```bash
python main.py
```

- The script performs the following:
  - Detects vehicles in the video.
  - Checks if a vehicle crosses during a red light (`semaforo == "ROJO"`).
  - If the vehicle crosses, extracts its license plate and prints the penalty message.

### 3. **Video Testing**
- Record a video using `test.py`:
  ```python
  film_video()
  ```
- Stream or replay a video:
  ```python
  play_video("path/to/video.avi")
  ```

### 4. **Object Tracking Recording**
To save a video recording of the object tracking results, use the script `tracking_record.py`:
```bash
python tracking_record.py
```
This script records the annotated video showing the tracked objects and their movements.

### 5. **Recording Annotated Video**
To save an annotated video of the object tracking using `grabar_video.py`:
```bash
python grabar_video.py
```
This script saves the tracked object output to `../data/video_final/output_video.avi` with bounding boxes, movement lines, and FPS display.

---

## **Key Functions**

### `calibration.py`
- `calibrar()`: Performs chessboard-based camera calibration.

### `tracking.py`
- `cruza(video_path)`: Tracks objects in the video and returns `True` if a vehicle crosses during a red light.

### `matric.py`
- `extract_license_plates(image)`: Detects and extracts license plate regions from an image.
- `extract_license_plate_text(plate)`: Extracts the text from a license plate image using Tesseract OCR.

### `test.py`
- `film_video()`: Records a video using the PiCamera.
- `stream_video()`: Streams the video feed live.
- `play_video(file_path)`: Plays a saved video.

### `tracking_record.py`
- Records and saves the video output with tracked objects, including bounding boxes and movement annotations.

### `grabar_video.py`
- `grabar_video_traking(video__path)`: Reads a video and saves the annotated object tracking results, including bounding boxes, movement center, and FPS overlay.

---

## **Example Output**

```bash
______EL SEMAFORO ESTA EN ROJO______
El vehículo va a ser multado
Se procede a analizar el video para obtener la matrícula . . .
El vehículo con matrícula: ABC1234 es multado con 600$ y 4 puntos en el carnet.
```

---

## **Configuration**

- Ensure the path to Tesseract is set correctly in your scripts:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

- Video files should be placed in the `../data/videos/` directory.

---

## **Usage Notes**
- The Kalman filter improves object tracking and helps mitigate issues with object occlusion.
- The license plate recognition may require fine-tuning based on the video quality and lighting conditions.

---

## **Future Improvements**
- Improve accuracy of license plate detection and OCR with additional image pre-processing.
- Support multi-camera inputs for broader coverage.
- Implement real-time alerts and automatic report generation.

---

Let me know if you'd like any edits or additions to this README!
