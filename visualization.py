import json
from typing import Dict, List

import cv2

from database import fetch_detections, fetch_interpolated_detections

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

MIN_LICENSE_SCORE = 0.2
RETENTION_FRAMES = 10


def _ensure_bbox(bbox) -> List[float]:
    if bbox in (None, ""):
        return []
    if isinstance(bbox, (list, tuple)):
        if len(bbox) == 4:
            return [float(v) for v in bbox]
        return []
    if isinstance(bbox, str):
        try:
            parsed = json.loads(bbox)
            if isinstance(parsed, list) and len(parsed) == 4:
                return [float(v) for v in parsed]
        except (ValueError, json.JSONDecodeError):
            pass
    return []


def _build_frame_lookup(rows: List[Dict]) -> Dict[int, Dict[int, Dict]]:
    frame_lookup: Dict[int, Dict[int, Dict]] = {}
    for row in rows:
        frame = int(row['frame_nmr'])
        car_id = int(row['car_id'])
        car_bbox = _ensure_bbox(row['car_bbox'])
        if not car_bbox:
            continue

        plate_bbox = _ensure_bbox(row.get('license_plate_bbox'))
        if not plate_bbox:
            plate_bbox = car_bbox

        entry = frame_lookup.setdefault(frame, {})
        entry[car_id] = {
            'car_bbox': car_bbox,
            'license_plate_bbox': plate_bbox,
            'license_plate_bbox_score': float(row.get('license_plate_bbox_score', 0.0) or 0.0),
            'license_number': (row.get('license_number') or '').strip(),
            'license_number_score': float(row.get('license_number_score', 0.0) or 0.0),
            'is_imputed': int(row.get('is_imputed', 0) or 0),
        }
    return frame_lookup


def visualize_results(video_path, conn, video_id, output_video_path, max_frames=None):
    interpolated_rows = fetch_interpolated_detections(conn, video_id)
    if interpolated_rows:
        frame_lookup = _build_frame_lookup(interpolated_rows)
    else:
        raw_rows = fetch_detections(conn, video_id)
        parsed_rows: List[Dict] = []
        for row in raw_rows:
            parsed_rows.append(
                {
                    'frame_nmr': int(row['frame_nmr']),
                    'car_id': int(row['car_id']),
                    'car_bbox': row['car_bbox'],
                    'license_plate_bbox': row['license_plate_bbox'],
                    'license_plate_bbox_score': row['license_plate_bbox_score'],
                    'license_number': row['license_number'],
                    'license_number_score': row['license_number_score'],
                    'is_imputed': 0,
                }
            )
        frame_lookup = _build_frame_lookup(parsed_rows)

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_nmr = -1
    ret = True
    last_known_license_plate: Dict[int, Dict] = {}

    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if max_frames is not None and frame_nmr >= max_frames:
            break
        if not ret:
            continue

        current_frame_data = frame_lookup.get(frame_nmr, {})

        for car_id, frame_entry in current_frame_data.items():
            frame_entry = dict(frame_entry)
            known_entry = last_known_license_plate.get(car_id, {})

            text = frame_entry.get('license_number', '')
            score = frame_entry.get('license_number_score', 0.0)

            if (not text or text == '0' or score < MIN_LICENSE_SCORE) and 'text' in known_entry:
                frame_entry['text'] = known_entry['text']
                frame_entry['text_score'] = known_entry.get('text_score', -1)
            elif text and text != '0' and score >= MIN_LICENSE_SCORE:
                frame_entry['text'] = text
                frame_entry['text_score'] = score

            if 'license_plate_bbox' not in frame_entry or not frame_entry['license_plate_bbox']:
                if 'license_plate_bbox' in known_entry:
                    frame_entry['license_plate_bbox'] = known_entry['license_plate_bbox']

            frame_entry['last_seen'] = frame_nmr
            last_known_license_plate[car_id] = {**known_entry, **frame_entry}

        stale_car_ids = [car_id for car_id, data in last_known_license_plate.items() if frame_nmr - data.get('last_seen', frame_nmr) > RETENTION_FRAMES]
        for car_id in stale_car_ids:
            last_known_license_plate.pop(car_id, None)

        render_candidates = []
        for car_id, data in last_known_license_plate.items():
            plate_text = str(data.get('text', '') or '').strip()
            if not plate_text or plate_text == '0':
                continue
            render_candidates.append((car_id, data, plate_text))

        if render_candidates:
            render_candidates.sort(key=lambda item: (item[1].get('text_score', -1), item[1].get('last_seen', -1)), reverse=True)
            _, render_data, plate_text = render_candidates[0]

            car_bbox = render_data.get('car_bbox')
            if car_bbox and len(car_bbox) == 4:
                car_x1, car_y1, car_x2, car_y2 = map(int, map(round, car_bbox))
                draw_border(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)

                plate_bbox = render_data.get('license_plate_bbox')
                if plate_bbox and len(plate_bbox) == 4:
                    x1, y1, x2, y2 = map(int, map(round, plate_bbox))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 12)

                (text_width, text_height), _ = cv2.getTextSize(
                    plate_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)

                background_height = text_height + 40
                text_bg_top_left = (int((car_x2 + car_x1 - text_width) / 2) - 20, int(car_y1 - background_height) - 40)
                text_bg_bottom_right = (int((car_x2 + car_x1 + text_width) / 2) + 20, int(car_y1) - 40)
                cv2.rectangle(frame, text_bg_top_left, text_bg_bottom_right, (255, 255, 255), -1)

                text_origin = (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - 60))
                cv2.putText(frame,
                            plate_text,
                            text_origin,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)

        out.write(frame)

    out.release()
    cap.release()
    return output_video_path