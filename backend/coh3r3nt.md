gatebil/
├── README.md
├── LICENSE
├── .gitignore
├── docker-compose.yml
├── .env.example
├── requirements/
│   ├── base.txt
│   ├── tensorflow.txt
│   ├── torch.txt
│   ├── api.txt
│   └── dev.txt
│
├── backend/
│   └── gatebil/
│       ├── main.py
│       ├── config.py
│       ├── routes/
│       │   ├── health.py
│       │   ├── verification.py
│       │   ├── liveness.py
│       │   └── document.py
│       │
│       ├── core/
│       │   ├── settings.py
│       │   ├── logging.py
│       │   ├── exceptions.py
│       │   └── security.py
│       │
│       ├── utils/
│       │   ├── image.py
│       │   ├── preprocessing.py
│       │   ├── geometry.py
│       │   ├── visualization.py
│       │   └── distance.py
│       │
│       ├── models/
│       │   ├── facenet/
│       │   │   ├── __init__.py
│       │   │   ├── models/
│       │   │   ├── weights/
│       │   │   └── utils/
│       │   │
│       │   ├── vggface/
│       │   │   ├── VGGFace.py
│       │   │   └── VGGFace2.py
│       │   │
│       │   ├── tensorflow/
│       │   │   ├── dsnt.py
│       │   │   ├── frozen_model.pb
│       │   │   └── inference.py
│       │   │
│       │   └── shared/
│       │       ├── loaders.py
│       │       ├── embeddings.py
│       │       └── inference_engine.py
│       │
│       ├── modules/
│       │   ├── document_verification/
│       │   │   ├── __init__.py
│       │   │   ├── detector.py
│       │   │   ├── warping.py
│       │   │   ├── extraction.py
│       │   │   ├── validation.py
│       │   │   └── pipeline.py
│       │   │
│       │   ├── face_verification/
│       │   │   ├── __init__.py
│       │   │   ├── face_verification.py
│       │   │   ├── embeddings.py
│       │   │   ├── matcher.py
│       │   │   └── pipeline.py
│       │   │
│       │   ├── liveness_detection/
│       │   │   ├── __init__.py
│       │   │   ├── blink_detection.py
│       │   │   ├── emotion_prediction.py
│       │   │   ├── face_orientation.py
│       │   │   ├── challenge_response.py
│       │   │   └── pipeline.py
│       │   │
│       │   └── ekyc/
│       │       ├── __init__.py
│       │       ├── orchestration.py
│       │       ├── session.py
│       │       ├── scoring.py
│       │       └── pipeline.py
│       │
│       ├── services/
│       │   ├── camera_service.py
│       │   ├── inference_service.py
│       │   ├── storage_service.py
│       │   ├── metrics_service.py
│       │   └── queue_service.py
│       │
│       ├── workers/
│       │   ├── worker.py
│       │   ├── queues.py
│       │   └── tasks/
│       │       ├── document_tasks.py
│       │       ├── face_tasks.py
│       │       └── liveness_tasks.py
│       │
│       ├── gui/
│       │   ├── __init__.py
│       │   ├── page1.py
│       │   ├── page2.py
│       │   ├── page3.py
│       │   └── utils.py
│       │
│       ├── templates/
│       │   └── index.html
│       │
│       ├── static/
│       │   ├── style.css
│       │   ├── script.js
│       │   ├── dl.svg
│       │   └── selfie.svg
│       │
│       ├── assets/
│       │   ├── samples/
│       │   │   ├── id-card.jpg
│       │   │   ├── ekyc.jpg
│       │   │   └── flow.jpg
│       │   │
│       │   ├── outputs/
│       │   │   ├── results.png
│       │   │   ├── cropped-warped.jpg
│       │   │   └── final.jpg
│       │   │
│       │   └── debug/
│       │       └── intermediate/
│       │
│       ├── notebooks/
│       │   ├── tensorflow_verification.ipynb
│       │   ├── liveness_experiments.ipynb
│       │   └── embedding_analysis.ipynb
│       │
│       └── tests/
│           ├── test_document.py
│           ├── test_face.py
│           ├── test_liveness.py
│           ├── test_orientation.py
│           ├── test_emotion.py
│           └── test_pipeline.py
│
├── frontend/
│   ├── Dockerfile
│   ├── Makefile
│   ├── go.mod
│   ├── go.sum
│   │
│   ├── cmd/
│   │   └── server/
│   │       └── main.go
│   │
│   ├── internal/
│   │   ├── config/
│   │   │   └── config.go
│   │   │
│   │   ├── domain/
│   │   │   └── kyc.go
│   │   │
│   │   ├── handler/
│   │   │   └── http.go
│   │   │
│   │   ├── repository/
│   │   │   └── postgres.go
│   │   │
│   │   ├── usecase/
│   │   │   └── kyc.go
│   │   │
│   │   ├── metrics/
│   │   │   └── metrics.go
│   │   │
│   │   └── worker/
│   │       └── pool.go
│   │
│   └── monitoring/
│       └── prometheus.yml
│
├── deployment/
│   ├── docker/
│   │   ├── backend.Dockerfile
│   │   ├── frontend.Dockerfile
│   │   └── nginx.conf
│   │
│   ├── k8s/
│   │   ├── backend.yaml
│   │   ├── frontend.yaml
│   │   ├── postgres.yaml
│   │   └── prometheus.yaml
│   │
│   └── scripts/
│       ├── bootstrap.sh
│       ├── migrate.sh
│       └── run-dev.sh
│
└── docs/
    ├── architecture.md
    ├── api.md
    ├── pipelines.md
    ├── models.md
    └── security.md
