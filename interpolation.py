import json
from typing import Dict, List

import numpy as np
from scipy.interpolate import interp1d

from database import fetch_detections, store_interpolated_detections

MAX_LICENSE_HOLD_FRAMES = 5

def _parse_bbox(raw: str, fallback: List[float]) -> List[float]:
    if not raw:
        return list(fallback)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) == 4:
            return [float(v) for v in parsed]
    except json.JSONDecodeError:
        pass
    return list(fallback)


def interpolate_bounding_boxes(data: List[Dict]) -> List[Dict]:
    interpolated_data: List[Dict] = []
    if not data:
        return interpolated_data

    cars: Dict[int, List[Dict]] = {}
    for row in data:
        cars.setdefault(row['car_id'], []).append(row)

    for car_id, rows in cars.items():
        rows.sort(key=lambda item: item['frame_nmr'])

        augmented_frames: List[int] = []
        augmented_car_bboxes: List[List[float]] = []
        augmented_plate_bboxes: List[List[float]] = []
        augmented_plate_scores: List[float] = []
        augmented_license_numbers: List[str] = []
        augmented_license_scores: List[float] = []
        augmented_is_imputed: List[int] = []

        last_known_license: str = ''
        last_license_frame: int = -1
        last_license_score: float = 0.0

        for row in rows:
            frame_number = int(row['frame_nmr'])
            car_bbox = np.array(row['car_bbox'], dtype=float)
            plate_bbox = np.array(row['license_plate_bbox'] or row['car_bbox'], dtype=float)

            if augmented_frames:
                prev_frame = augmented_frames[-1]
                if frame_number - prev_frame > 1:
                    prev_car_bbox = np.array(augmented_car_bboxes[-1], dtype=float)
                    prev_plate_bbox = np.array(augmented_plate_bboxes[-1], dtype=float)
                    x = np.array([prev_frame, frame_number])
                    x_new = np.arange(prev_frame + 1, frame_number)
                    car_interp = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    plate_interp = interp1d(x, np.vstack((prev_plate_bbox, plate_bbox)), axis=0, kind='linear')

                    for missing_frame in x_new:
                        missing_frame = int(missing_frame)
                        car_box = car_interp(missing_frame).astype(float).tolist()
                        plate_box = plate_interp(missing_frame).astype(float).tolist()

                        if last_license_frame != -1 and missing_frame - last_license_frame <= MAX_LICENSE_HOLD_FRAMES and last_known_license:
                            license_number = last_known_license
                            license_score = last_license_score
                        else:
                            license_number = '0'
                            license_score = 0.0

                        augmented_frames.append(missing_frame)
                        augmented_car_bboxes.append(car_box)
                        augmented_plate_bboxes.append(plate_box)
                        augmented_plate_scores.append(0.0)
                        augmented_license_numbers.append(license_number)
                        augmented_license_scores.append(license_score)
                        augmented_is_imputed.append(1)

            license_number = row['license_number'] or '0'
            license_score = float(row['license_number_score'] or 0.0)
            plate_score = float(row['license_plate_bbox_score'] or 0.0)

            if license_number and license_number != '0':
                last_known_license = license_number
                last_license_frame = frame_number
                last_license_score = license_score
            elif last_license_frame != -1 and frame_number - last_license_frame <= MAX_LICENSE_HOLD_FRAMES and last_known_license:
                license_number = last_known_license
                license_score = last_license_score
            else:
                license_number = '0'
                license_score = 0.0

            augmented_frames.append(frame_number)
            augmented_car_bboxes.append(car_bbox.astype(float).tolist())
            augmented_plate_bboxes.append(plate_bbox.astype(float).tolist())
            augmented_plate_scores.append(plate_score)
            augmented_license_numbers.append(license_number)
            augmented_license_scores.append(license_score)
            augmented_is_imputed.append(0)

        for frame, car_box, plate_box, plate_score, lic_number, lic_score, is_imputed in zip(
            augmented_frames,
            augmented_car_bboxes,
            augmented_plate_bboxes,
            augmented_plate_scores,
            augmented_license_numbers,
            augmented_license_scores,
            augmented_is_imputed,
        ):
            interpolated_data.append(
                {
                    'frame_nmr': frame,
                    'car_id': car_id,
                    'car_bbox': car_box,
                    'license_plate_bbox': plate_box,
                    'license_plate_bbox_score': plate_score,
                    'license_number': lic_number,
                    'license_number_score': lic_score,
                    'is_imputed': is_imputed,
                }
            )

    interpolated_data.sort(key=lambda row: (row['frame_nmr'], row['car_id']))
    return interpolated_data


def interpolate_results(conn, video_id: str) -> int:
    raw_rows = fetch_detections(conn, video_id)
    if not raw_rows:
        with conn:
            conn.execute("DELETE FROM interpolated_detections WHERE video_id = ?", (video_id,))
        return 0

    parsed_rows: List[Dict] = []
    for row in raw_rows:
        car_bbox = _parse_bbox(row['car_bbox'], [0.0, 0.0, 0.0, 0.0])
        plate_bbox = _parse_bbox(row['license_plate_bbox'], car_bbox)
        parsed_rows.append(
            {
                'frame_nmr': int(row['frame_nmr']),
                'car_id': int(row['car_id']),
                'car_bbox': car_bbox,
                'license_plate_bbox': plate_bbox,
                'license_plate_bbox_score': float(row['license_plate_bbox_score'] or 0.0),
                'license_number': row['license_number'] or '0',
                'license_number_score': float(row['license_number_score'] or 0.0),
            }
        )

    with conn:
        conn.execute("DELETE FROM interpolated_detections WHERE video_id = ?", (video_id,))

    interpolated_data = interpolate_bounding_boxes(parsed_rows)
    if not interpolated_data:
        return 0

    store_interpolated_detections(conn, video_id, interpolated_data)
    return len(interpolated_data)
