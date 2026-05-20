# Gatebil

> A modular biometric identity verification and eKYC platform combining
> face verification, liveness detection, document analysis, and scalable backend orchestration.

---

# Overview

Gatebil appears to be an experimental yet ambitious end-to-end electronic Know Your Customer (eKYC) platform focused on biometric identity verification.

The repository combines multiple subsystems:

- Facial verification
- Liveness detection
- ID card analysis
- Emotion and blink detection
- Face orientation validation
- TensorFlow-based inference pipelines
- A Go orchestration backend
- Monitoring and worker infrastructure
- A lightweight frontend/API layer

The project architecture suggests a system intended for:

- onboarding users remotely
- identity validation
- fraud prevention
- anti-spoofing
- fintech or mobility verification flows
- secure authentication pipelines

The repo feels like a fusion of:

- machine vision laboratory
- production service mesh
- experimental biometric sandbox

A small digital customs checkpoint with neural networks guarding the gates.

---

# Repository Structure

```text
.
├── backend/
│   ├── bil/        # Document + TensorFlow verification pipeline
│   └── gate/       # Facial verification and liveness detection
│
├── frontend/       # Go backend/API orchestration layer
│
└── monitoring/     # Prometheus monitoring


---

Core Components

1. backend/bil

The bil service appears focused on:

ID card extraction

document alignment

TensorFlow inference

image preprocessing

verification workflows


Key Files

File	Purpose

app.py	Main Flask or Python application
frozen_model.pb	Pretrained TensorFlow frozen graph
dsnt.py	Deep Spatial Neural Transform utilities
Tensorflow - Verification.ipynb	Experimental notebook for model evaluation
results.png	Verification output visualization


Observations

The presence of:

warped image outputs

cropped image intermediates

TensorFlow graphs

DSNT spatial transforms


suggests the pipeline may:

1. detect ID boundaries


2. correct perspective distortion


3. crop regions of interest


4. run verification or classification models



This resembles classic document normalization pipelines used in OCR and ID verification systems.

Example Workflow

User Upload
    ↓
Document Detection
    ↓
Perspective Warp
    ↓
Region Extraction
    ↓
TensorFlow Inference
    ↓
Verification Result


---

2. backend/gate

This is the biometric engine of the repository.

It appears responsible for:

face verification

anti-spoofing

liveness detection

challenge-response validation

facial embedding generation


This subsystem is significantly more mature and modular.


---

Face Verification

Files

face_verification.py
verification_models/
facenet/
utils/distance.py

The repository includes:

FaceNet

VGGFace

VGGFace2

MTCNN detection

embedding distance utilities


This indicates support for:

facial embedding generation

cosine/euclidean similarity matching

identity comparison workflows


Likely flow:

Selfie
    ↓
Face Detection (MTCNN)
    ↓
Embedding Generation
    ↓
Distance Calculation
    ↓
Identity Match Score


---

Liveness Detection

Files

blink_detection.py
emotion_prediction.py
face_orientation.py
challenge_response.py

This section strongly suggests active anti-spoofing defenses.

The platform likely validates that the subject is:

physically present

responsive

not using a photograph/video replay


Supported Checks

Capability	Purpose

Blink Detection	Detect live eye motion
Emotion Prediction	Challenge-response verification
Face Orientation	Head movement tracking
Challenge Response	Dynamic anti-spoof interaction


This architecture resembles the systems used in:

digital banking onboarding

crypto exchange KYC

secure remote verification

high-trust authentication systems



---

GUI Components

gui/page1.py
gui/page2.py
gui/page3.py

The Python GUI suggests an interactive verification flow.

Possibly:

1. Upload ID


2. Capture selfie


3. Perform liveness challenge



The presence of flow.jpg and ekyc.jpg supports this interpretation.


---

Testing Infrastructure

tests/

Includes:

blink tests

emotion tests

orientation tests


This is a good indicator the author intended repeatable biometric validation behavior rather than purely experimental notebooks.


---

3. Frontend (Go Backend)

Interestingly, the frontend/ directory is not a traditional frontend.

It appears to be a Go orchestration/API service.


---

Architecture

cmd/server/main.go
internal/

This follows idiomatic Go clean architecture patterns.

Layers

Layer	Purpose

config	Environment/config management
domain	Core business models
handler	HTTP transport layer
repository	Persistence/database layer
usecase	Business logic
worker	Concurrent task execution
metrics	Monitoring instrumentation


This suggests the project evolved beyond a prototype into something approaching production infrastructure.


---

Features Implied

Worker Pool

internal/worker/pool.go

Likely used for:

async verification jobs

image processing queues

concurrent inference execution


PostgreSQL Repository

repository/postgres.go

Implies persistence for:

KYC sessions

verification results

audit trails

user onboarding records


Metrics

metrics.go
prometheus.yml

Monitoring support indicates operational awareness.

The system was likely intended to expose:

request latency

verification success rate

worker queue depth

inference timings



---

Technologies Used

Machine Learning / Vision

TensorFlow

FaceNet

VGGFace

MTCNN

OpenCV

Deep learning embeddings


Backend

Go

Flask/Python

PostgreSQL

Docker

Prometheus


Infrastructure

Docker Compose

Worker pools

Monitoring stack

Modular service separation



---

Likely End-to-End Flow

User Uploads ID
        ↓
Document Verification
        ↓
Selfie Capture
        ↓
Face Detection
        ↓
Liveness Challenge
        ↓
Embedding Comparison
        ↓
Verification Decision
        ↓
Persist Results
        ↓
Expose Metrics


---

Design Philosophy

The repository demonstrates a strong emphasis on:

modularity

experimentation

biometric validation

fraud resistance

production-oriented architecture


There is an interesting contrast between:

research-style ML experimentation (.ipynb, TensorFlow models)

and structured backend engineering (internal/usecase, worker pools, metrics)


The result feels like a bridge between:

> a computer vision research lab
and
a fintech onboarding platform.




---

Potential Use Cases

Remote KYC onboarding

Banking verification

Crypto exchange identity validation

Driver/passenger verification

Secure account recovery

Fraud-resistant authentication

Digital identity systems



---

Security Considerations

Because this repository processes biometric and identity data, production deployments would likely require:

encryption at rest

secure image handling

GDPR/privacy compliance

audit logging

model hardening

anti-replay protections

secure storage of embeddings



---

Future Enhancements (Speculative)

Based on the architecture, future evolution could include:

OCR extraction pipelines

NFC passport support

WebRTC live capture

distributed inference workers

GPU acceleration

Kubernetes deployment

multi-model ensemble verification

risk scoring engines



---

Conclusion

Gatebil appears to be an advanced biometric verification platform exploring the intersection of:

computer vision

identity verification

backend orchestration

fraud prevention

scalable service architecture


The repository combines experimental machine learning systems with operational backend engineering in a way that suggests a long-term vision beyond simple demos.

It is simultaneously:

a biometric laboratory

a verification gateway

and an infrastructure scaffold for digital identity systems.



