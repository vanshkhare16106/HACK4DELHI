import cv2
from detection.vehicle_detector import VehicleDetector
from detection.LineCrossing import LineCrossing
from detection.plate_reader import PlateReader
from logic.parking_manager import ParkingManager
from logic.counter import VehicleCounter  # <--- ✅ ADDED THIS BACK


def main():
    # -------- CONFIG --------
    MODEL_PATH = "models/yolov8n.pt"
    MAX_CAPACITY = 10  # Set your max parking spots here

    # -------- INITIALIZE --------
    detector = VehicleDetector(MODEL_PATH)
    plate_reader = PlateReader()
    manager = ParkingManager(hourly_rate=20)
    counter = VehicleCounter(MAX_CAPACITY)  # <--- ✅ INITIALIZE COUNTER

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Error: Cannot open video")
        return

    # Line Setup
    ret, frame = cap.read()
    if not ret: return
    height, width, _ = frame.shape
    line_y = height // 2
    line_cross = LineCrossing(line_y)

    print("✅ System Ready. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        detections = detector.detect(frame)

        # Draw Line
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

        for (x1, y1, x2, y2, label, conf, track_id) in detections:
            cy = int((y1 + y2) / 2)
            cx = int((x1 + x2) / 2)

            # Check for Crossing Event
            event = line_cross.check(track_id, cy)

            if event:
                # 1. Update Counter
                counter.process_event(event)  # <--- ✅ UPDATE COUNT

                # 2. Trigger OCR
                print(f"👀 Scanning Plate for {event}...")
                plate_text = plate_reader.read_plate(frame, x1, y1, x2, y2)

                # 3. Manage Ticket/Fee
                msg = ""
                if event == "ENTRY":
                    msg = manager.register_entry(plate_text)
                elif event == "EXIT":
                    msg = manager.register_exit(plate_text)

                print(f"📢 {msg}")

                # Visual Feedback for Event
                cv2.putText(frame, msg, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Draw normal bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ✅ DISPLAY LIVE COUNT ON SCREEN
        count_text = f"Parking: {counter.get_count()} / {MAX_CAPACITY}"
        # Change color to RED if full, GREEN otherwise
        color = (0, 0, 255) if counter.get_count() >= MAX_CAPACITY else (0, 255, 0)

        cv2.putText(frame, count_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Smart Parking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()