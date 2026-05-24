# app/storage/temp.py

from pathlib import Path
from uuid import uuid4

import cv2


TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)


def write_temp_image(image):

    path = TMP_DIR / f"{uuid4().hex}.jpg"

    cv2.imwrite(str(path), image)

    return str(path)

