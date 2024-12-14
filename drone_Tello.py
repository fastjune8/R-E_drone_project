from djitellopy import Tello
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO('yolov8x.pt')  # Use the yolov8x model

# Drone parameters
width = 640
height = 480
death_zone = 100

# Initialize Tello drone
drone = Tello()
drone.connect()
print(f"Battery: {drone.get_battery()}%")

drone.streamoff()
drone.streamon()

def track_object(results, img):
    """Track the largest detected object and determine drone movements."""
    global direction
    direction = 0  # 0: No movement, 1: Left, 2: Right, 3: Up, 4: Down

    # Parse detection results
    detections = results[0].boxes.data.cpu().numpy()  # YOLOv8 detection results
    if len(detections) == 0:
        return img  # No detections

    largest_box = None
    largest_area = 0

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        class_id = int(cls)
        
        # Specify target class (e.g., 0 for person)
        target_class = 0
        if class_id != target_class:
            continue

        # Calculate bounding box area
        area = (x2 - x1) * (y2 - y1)
        if area > largest_area:
            largest_area = area
            largest_box = (int(x1), int(y1), int(x2), int(y2))

    if largest_box is None:
        return img  # No target object detected

    x1, y1, x2, y2 = largest_box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Draw bounding box and center point
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    # Determine movement direction
    if cx < width // 2 - death_zone:
        direction = 1  # Move left
    elif cx > width // 2 + death_zone:
        direction = 2  # Move right
    elif cy < height // 2 - death_zone:
        direction = 3  # Move up
    elif cy > height // 2 + death_zone:
        direction = 4  # Move down
    else:
        direction = 0  # Stay

    return img

def send_movement_commands():
    """Send movement commands to the drone based on direction."""
    global direction

    # Reset velocities
    drone.for_back_velocity = 0
    drone.left_right_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0

    # Set velocities based on direction
    if direction == 1:
        drone.left_right_velocity = -20  # Move left
    elif direction == 2:
        drone.left_right_velocity = 20  # Move right
    elif direction == 3:
        drone.up_down_velocity = 20  # Move up
    elif direction == 4:
        drone.up_down_velocity = -20  # Move down

    # Send command to the drone
    drone.send_rc_control(drone.left_right_velocity, drone.for_back_velocity,
                          drone.up_down_velocity, drone.yaw_velocity)

try:
    while True:
        # Get video frame from drone
        frame = drone.get_frame_read().frame
        frame = cv2.resize(frame, (width, height))

        # Perform object detection
        results = model.predict(source=frame, conf=0.5, save=False, imgsz=640, verbose=False)

        # Track object and annotate frame
        annotated_frame = track_object(results, frame)

        # Display the frame
        cv2.imshow("Drone Camera", annotated_frame)

        # Send movement commands to the drone
        send_movement_commands()

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExiting program...")

finally:
    drone.land()
    cv2.destroyAllWindows()

