import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from config import PROCESS_FIRST_N_SECONDS
from database import store_detections
from util import get_car, read_license_plate


def process_video(video_path, progress_bar, conn, video_id):
    mot_tracker = DeepSort(max_age=30)
    coco_model = YOLO('./models/yolov8n.pt')
    license_plate_detector = YOLO('./models/license_plate_detector.pt')
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    max_frames = None
    if PROCESS_FIRST_N_SECONDS and PROCESS_FIRST_N_SECONDS > 0:
        max_frames = int(fps * PROCESS_FIRST_N_SECONDS)
        total_frames = min(total_frames, max_frames)

    vehicles = [2, 3, 5, 7]
    frame_nmr = -1
    ret = True
    detection_count = 0
    last_known_license_plate = {}
    while ret:
        frame_nmr += 1
        if max_frames is not None and frame_nmr >= max_frames:
            break
        ret, frame = cap.read()
        if ret:
            progress_bar.progress(frame_nmr / total_frames, text=f"Detecting vehicles and license plates in frame {frame_nmr}/{total_frames}")
            frame_results = {}
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append((([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], score, 'car')))
            
            tracks = mot_tracker.update_tracks(detections_, frame=frame)
            track_ids = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                track_ids.append([ltrb[0], ltrb[1], ltrb[2], ltrb[3], track_id])

            # Preprocess frame for license plate detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equ_frame = cv2.equalizeHist(gray_frame)
            license_plates_frame = cv2.cvtColor(equ_frame, cv2.COLOR_GRAY2BGR)

            license_plates = license_plate_detector(license_plates_frame, conf=0.5)[0]
            frame_height, frame_width = frame.shape[:2]
            active_car_ids = {track[4] for track in track_ids}
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, np.array(track_ids))
                if car_id == -1:
                    continue

                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(frame_width, int(x2)), min(frame_height, int(y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                license_plate_crop = frame[y1:y2, x1:x2, :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                license_plate_crop_gray = cv2.GaussianBlur(license_plate_crop_gray, (3, 3), 0)
                _, license_plate_crop_thresh = cv2.threshold(
                    license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                entry = last_known_license_plate.get(car_id, {})
                entry['bbox'] = [x1, y1, x2, y2]
                entry['bbox_score'] = score
                entry['last_seen_frame'] = frame_nmr

                if license_plate_text is not None:
                    best_score = entry.get('text_score', -1)
                    if license_plate_text_score >= best_score:
                        entry['text'] = license_plate_text
                        entry['text_score'] = license_plate_text_score

                last_known_license_plate[car_id] = entry

            for car_id in list(last_known_license_plate.keys()):
                if car_id not in active_car_ids:
                    last_seen = last_known_license_plate[car_id].get('last_seen_frame', -1)
                    if frame_nmr - last_seen > 10:
                        last_known_license_plate.pop(car_id, None)
                    continue

                plate_entry = last_known_license_plate[car_id]
                if 'text' not in plate_entry:
                    continue

                xcar1, ycar1, xcar2, ycar2, _ = next(track for track in track_ids if track[4] == car_id)
                plate_bbox = plate_entry.get('bbox', [xcar1, ycar1, xcar2, ycar2])
                frame_results[car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': plate_bbox,
                        'text': plate_entry['text'],
                        'bbox_score': plate_entry.get('bbox_score', 0),
                        'text_score': plate_entry.get('text_score', 0)
                    }
                }
            if frame_results:
                detection_count += len(frame_results)
                store_detections(conn, video_id, {frame_nmr: frame_results})
    cap.release()
    return max_frames, detection_count
