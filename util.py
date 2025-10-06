import string
import re
import itertools
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

INDIAN_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ', 'HP', 'HR',
    'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB',
    'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
}

INDIAN_PLATE_PATTERNS = [
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$'),
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{3}$'),
    re.compile(r'^[A-Z]{2}\d{1}[A-Z]{1,2}\d{4}$')
]

PLATE_TEMPLATES = {
    10: ['A', 'A', 'D', 'D', 'A', 'A', 'D', 'D', 'D', 'D'],
    9: ['A', 'A', 'D', 'A', 'A', 'D', 'D', 'D', 'D'],
    8: ['A', 'A', 'D', 'D', 'D', 'D', 'D', 'D']
}

STATE_DIGIT_TO_LETTER = {
    '0': ['O', 'D', 'Q'],
    '1': ['I', 'L'],
    '2': ['Z'],
    '3': ['B'],
    '4': ['A'],
    '5': ['S'],
    '6': ['G'],
    '7': ['T'],
    '8': ['B'],
    '9': ['G', 'Q']
}

LETTER_TO_DIGIT = {
    **dict_char_to_int,
    'B': '8',
    'D': '0',
    'Q': '0',
    'T': '7',
    'Z': '2',
    'L': '1'
}


def _fix_state_prefix(text):
    if len(text) < 2:
        return text

    prefix = text[:2]
    upper_prefix = prefix.upper()
    if upper_prefix in INDIAN_STATE_CODES:
        return upper_prefix + text[2:]

    options = []
    for ch in prefix:
        candidates = {ch.upper()}
        if ch.isdigit():
            candidates.update(STATE_DIGIT_TO_LETTER.get(ch, []))
        options.append(sorted(candidates))

    for combo in itertools.product(*options):
        candidate_prefix = ''.join(combo)
        if candidate_prefix in INDIAN_STATE_CODES:
            return candidate_prefix + text[2:]

    return upper_prefix + text[2:]


def _enforce_indian_structure(text):
    if not text:
        return text

    text = _fix_state_prefix(text)
    chars = list(text)

    for idx in range(2, min(4, len(chars))):
        ch = chars[idx]
        if ch.isalpha() and ch in LETTER_TO_DIGIT:
            chars[idx] = LETTER_TO_DIGIT[ch]

    if len(chars) > 4:
        ch = chars[4]
        if ch.isdigit() and ch in dict_int_to_char:
            chars[4] = dict_int_to_char[ch]

    return ''.join(chars).upper()


def evaluate_plate_candidate(text):
    normalized = ''.join(ch for ch in text if ch.isalnum()).upper()
    if len(normalized) < 6 or len(normalized) > 12:
        return False, -1

    if not any(ch.isalpha() for ch in normalized):
        return False, -1

    if not any(ch.isdigit() for ch in normalized):
        return False, -1

    allowed_chars = set(string.ascii_uppercase + string.digits)
    if not all(ch in allowed_chars for ch in normalized):
        return False, -1

    state_bonus = 12 if normalized[:2] in INDIAN_STATE_CODES else 0

    for weight, pattern in enumerate(INDIAN_PLATE_PATTERNS, start=1):
        if pattern.match(normalized):
            return True, 100 - weight + state_bonus

    template = PLATE_TEMPLATES.get(len(normalized))
    score = 0
    if template:
        for idx, expected_type in enumerate(template):
            ch = normalized[idx]
            if expected_type == 'A':
                score += 3 if ch.isalpha() else -2
            else:
                score += 3 if ch.isdigit() else -2

    score += sum(1 for ch in normalized if ch.isalpha())
    score += sum(1 for ch in normalized if ch.isdigit())

    return True, score + state_bonus


def license_complies_format(text):
    """
    Check if the license plate text complies with expected alphanumeric formats.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """

    is_valid, _ = evaluate_plate_candidate(text.upper())
    return is_valid


def format_license(text):
    """
    Provide a normalized representation of the license plate text.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """

    return text.upper()


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    best_candidate = None
    best_score = -1
    best_detection_score = 0

    for detection in detections:
        bbox, text, detection_score = detection

        normalized = text.upper().replace(' ', '').replace('-', '')
        normalized = ''.join(ch for ch in normalized if ch.isalnum())

        candidates = [normalized]
        candidates.append(''.join(dict_char_to_int.get(ch, ch) for ch in normalized))
        candidates.append(''.join(dict_int_to_char.get(ch, ch) for ch in normalized))
        candidates.append(''.join(dict_int_to_char.get(dict_char_to_int.get(ch, ch), dict_char_to_int.get(ch, ch)) for ch in normalized))

        refined_candidates = []
        for candidate in dict.fromkeys(candidates):
            if candidate:
                refined_candidates.append(_enforce_indian_structure(candidate.upper()))

        for candidate in dict.fromkeys(refined_candidates):
            if not candidate:
                continue

            is_valid, candidate_score = evaluate_plate_candidate(candidate.upper())
            if not is_valid:
                continue

            total_score = candidate_score + detection_score
            if total_score > best_score:
                best_candidate = candidate
                best_score = total_score
                best_detection_score = detection_score

    if best_candidate is not None:
        return format_license(best_candidate), best_detection_score

    return None, None

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        # Check for intersection between license plate and car bounding boxes
        if float(x1) < float(xcar2) and float(x2) > float(xcar1) and float(y1) < float(ycar2) and float(y2) > float(ycar1):
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
