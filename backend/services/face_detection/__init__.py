# __init__.py
"""
face_detection
==============

Face detection and embedding pipeline.

Quick start
-----------
    from face_detection import FaceDetector

    detector = FaceDetector(device='cuda', pretrained='vggface2')

    result = detector.detect_and_embed(img)
    result.embeddings   # (N, 512) L2-normalized
    result.boxes        # (N, 4)
    result.confidence   # (N,) detection probabilities

Lower-level access
------------------
    from face_detection.mtcnn import MTCNN
    from face_detection.inception_resnet_v1 import InceptionResnetV1
"""

from .face_detection import FaceDetector, DetectionResult, EmbeddingResult
from .mtcnn import MTCNN, PNet, RNet, ONet
from .inception_resnet_v1 import (
    InceptionResnetV1,
    fixed_image_standardization,
    prewhiten,
)

__all__ = [
    'FaceDetector',
    'DetectionResult',
    'EmbeddingResult',
    'MTCNN',
    'PNet',
    'RNet',
    'ONet',
    'InceptionResnetV1',
    'fixed_image_standardization',
    'prewhiten',
]
