from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

import io
import os
import uuid
import threading
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    efficientnet_b3,
    resnet50
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "app", "models")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("PROJECT QUANTUM API")
print("=" * 70)
print("Device:", device)
print("Model directory:", MODEL_DIR)


# ============================================================
# ASYNC PREDICTION STORAGE
# ============================================================

prediction_jobs = {}
prediction_lock = threading.Lock()


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="PROJECT QUANTUM API",
    description="Hybrid Quantum-Classical Early Disease Detection Platform",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# EFFICIENTNET-B0
# ============================================================

print("\n" + "=" * 70)
print("LOADING EFFICIENTNET-B0")
print("=" * 70)

b0 = efficientnet_b0(weights=None)

b0.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, 2)
)

b0_checkpoint = torch.load(
    os.path.join(
        MODEL_DIR,
        "efficientnet_b0_best.pth"
    ),
    map_location=device,
    weights_only=False
)

if "model_state_dict" in b0_checkpoint:
    b0.load_state_dict(
        b0_checkpoint["model_state_dict"],
        strict=True
    )
else:
    b0.load_state_dict(
        b0_checkpoint,
        strict=True
    )

b0 = b0.to(device)
b0.eval()

print("EfficientNet-B0 loaded successfully.")
print(
    "Validation AUC:",
    b0_checkpoint.get("val_auc")
)


# ============================================================
# RESNET-50
# ============================================================

print("\n" + "=" * 70)
print("LOADING RESNET-50")
print("=" * 70)

resnet = resnet50(weights=None)

resnet.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(2048, 2)
)

resnet_checkpoint = torch.load(
    os.path.join(
        MODEL_DIR,
        "resnet50_best.pth"
    ),
    map_location=device,
    weights_only=False
)

if "model_state_dict" in resnet_checkpoint:
    resnet.load_state_dict(
        resnet_checkpoint["model_state_dict"],
        strict=True
    )
else:
    resnet.load_state_dict(
        resnet_checkpoint,
        strict=True
    )

resnet = resnet.to(device)
resnet.eval()

print("ResNet-50 loaded successfully.")
print(
    "Validation AUC:",
    resnet_checkpoint.get("val_auc")
)


# ============================================================
# EFFICIENTNET-B3
# ============================================================

print("\n" + "=" * 70)
print("LOADING EFFICIENTNET-B3")
print("=" * 70)

b3 = efficientnet_b3(weights=None)

b3.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1536, 2)
)

b3_checkpoint = torch.load(
    os.path.join(
        MODEL_DIR,
        "efficientnet_b3_backbone.pth"
    ),
    map_location=device,
    weights_only=False
)

if "model_state_dict" in b3_checkpoint:
    b3.load_state_dict(
        b3_checkpoint["model_state_dict"],
        strict=True
)
else:
    b3.load_state_dict(
        b3_checkpoint,
        strict=True
)

# Remove classifier because B3 is used as feature extractor
b3.classifier = nn.Identity()

b3 = b3.to(device)
b3.eval()

print("EfficientNet-B3 backbone loaded successfully.")


# ============================================================
# SCALER
# ============================================================

print("\n" + "=" * 70)
print("LOADING CNN FEATURE SCALER")
print("=" * 70)

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "cnn_feature_scaler.pkl"
    )
)

print("CNN feature scaler loaded successfully.")


# ============================================================
# SUPERVISED PROJECTION
#
# 1536 → 128 → 32 → 8
# ============================================================

print("\n" + "=" * 70)
print("LOADING 8D SUPERVISED PROJECTION")
print("=" * 70)

projection = nn.Sequential(

    nn.Linear(1536, 128),

    nn.BatchNorm1d(128),

    nn.ReLU(),

    nn.Dropout(0.2),

    nn.Linear(128, 32),

    nn.BatchNorm1d(32),

    nn.ReLU(),

    nn.Linear(32, 8)
)

projection_checkpoint = torch.load(
    os.path.join(
        MODEL_DIR,
        "supervised_projection_8d_NEW.pth"
    ),
    map_location=device,
    weights_only=False
)

if "model_state_dict" in projection_checkpoint:

    projection.load_state_dict(
        projection_checkpoint["model_state_dict"],
        strict=True
    )

else:

    projection.load_state_dict(
        projection_checkpoint,
        strict=True
    )

projection = projection.to(device)
projection.eval()

print("8D supervised projection loaded successfully.")


# ============================================================
# SAFE VQC
# ============================================================

print("\n" + "=" * 70)
print("INITIALIZING 8-QUBIT VQC")
print("=" * 70)


class SafeVQC(nn.Module):

    def __init__(
        self,
        n_qubits=8,
        n_layers=3
    ):

        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.q_weights = nn.Parameter(
            torch.randn(
                n_layers,
                n_qubits,
                2
            ) * 0.05
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                n_qubits,
                16
            ),

            nn.Tanh(),

            nn.Linear(
                16,
                2
            )
        )


    def ry(self, theta):

        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)

        return torch.stack(
            [
                torch.stack(
                    [c, -s],
                    dim=-1
                ),

                torch.stack(
                    [s, c],
                    dim=-1
                )
            ],
            dim=-2
        ).to(torch.complex64)


    def rz(self, theta):

        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)

        zero = torch.zeros_like(c)

        return torch.stack(
            [

                torch.stack(
                    [
                        torch.complex(c, -s),
                        zero
                    ],
                    dim=-1
                ),

                torch.stack(
                    [
                        zero,
                        torch.complex(c, s)
                    ],
                    dim=-1
                )

            ],
            dim=-2
        )


    def _apply_single_qubit(
        self,
        state,
        gate,
        qubit
    ):

        batch = state.shape[0]

        tensor = state.reshape(
            batch,
            *([2] * self.n_qubits)
        )

        tensor = tensor.movedim(
            qubit + 1,
            -1
        )

        tensor = torch.matmul(
            tensor,
            gate.transpose(-1, -2)
        )

        tensor = tensor.movedim(
            -1,
            qubit + 1
        )

        return tensor.reshape(
            batch,
            -1
        )


    def _apply_cnot(
        self,
        state,
        control,
        target
    ):

        result = state.clone()

        dim = 2 ** self.n_qubits

        for i in range(dim):

            if ((i >> control) & 1) == 1:

                flipped = i ^ (1 << target)

                if i < flipped:

                    a = state[:, i].clone()
                    b = state[:, flipped].clone()

                    result[:, i] = b
                    result[:, flipped] = a

        return result


    def forward(self, x):

        batch = x.shape[0]

        state = torch.zeros(
            batch,
            2 ** self.n_qubits,
            dtype=torch.complex64,
            device=x.device
        )

        state[:, 0] = 1.0


        # ----------------------------------------------------
        # Angle embedding
        # ----------------------------------------------------

        for q in range(self.n_qubits):

            state = self._apply_single_qubit(
                state,
                self.ry(x[:, q]),
                q
            )


        # ----------------------------------------------------
        # Variational layers
        # ----------------------------------------------------

        for layer in range(
            self.n_layers
        ):

            for q in range(
                self.n_qubits
            ):

                state = self._apply_single_qubit(
                    state,

                    self.ry(
                        self.q_weights[
                            layer,
                            q,
                            0
                        ].expand(batch)
                    ),

                    q
                )

                state = self._apply_single_qubit(
                    state,

                    self.rz(
                        self.q_weights[
                            layer,
                            q,
                            1
                        ].expand(batch)
                    ),

                    q
                )


            # CNOT ring

            for q in range(
                self.n_qubits
            ):

                state = self._apply_cnot(
                    state,
                    q,
                    (q + 1) % self.n_qubits
                )


        # ----------------------------------------------------
        # Measurement
        # ----------------------------------------------------

        probs = torch.abs(state) ** 2

        expectations = []

        for q in range(
            self.n_qubits
        ):

            values = torch.tensor(
                [
                    1.0
                    if ((i >> q) & 1) == 0
                    else -1.0

                    for i in range(
                        2 ** self.n_qubits
                    )
                ],
                device=x.device
            )

            expectations.append(
                torch.sum(
                    probs * values,
                    dim=1
                )
            )

        z = torch.stack(
            expectations,
            dim=1
        )

        return self.classifier(z)


# ============================================================
# HYBRID MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING HYBRID B3 + 8-QUBIT VQC")
print("=" * 70)


class HybridModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.vqc = SafeVQC(
            n_qubits=8,
            n_layers=3
        )

        self.classical = nn.Sequential(

            nn.Linear(
                8,
                16
            ),

            nn.ReLU(),

            nn.Dropout(
                0.15
            ),

            nn.Linear(
                16,
                2
            )
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                4,
                16
            ),

            nn.ReLU(),

            nn.Dropout(
                0.15
            ),

            nn.Linear(
                16,
                2
            )
        )


    def forward(self, x):

        q_logits = self.vqc(x)

        c_logits = self.classical(x)

        fusion_input = torch.cat(
            [
                q_logits,
                c_logits
            ],
            dim=1
        )

        return self.fusion(
            fusion_input
        )


hybrid = HybridModel()

hybrid_checkpoint = torch.load(
    os.path.join(
        MODEL_DIR,
        "hybrid_safe_vqc_FINAL.pth"
    ),
    map_location=device,
    weights_only=False
)

if "model_state_dict" in hybrid_checkpoint:

    hybrid.load_state_dict(
        hybrid_checkpoint["model_state_dict"],
        strict=True
    )

else:

    hybrid.load_state_dict(
        hybrid_checkpoint,
        strict=True
    )

hybrid = hybrid.to(device)
hybrid.eval()

print("Hybrid B3 + 8-Qubit VQC loaded successfully.")


# ============================================================
# IMAGE → 8D
# ============================================================

def image_to_8d(image):

    image = image.convert("RGB")

    x = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        features = b3(x)

        scaled = scaler.transform(
            features.cpu().numpy()
        )

        scaled = torch.tensor(
            scaled,
            dtype=torch.float32,
            device=device
        )

        features_8d = projection(
            scaled
        )

    return features_8d


# ============================================================
# CLASSICAL MODEL PREDICTION
# ============================================================

def predict_classical_model(
    model,
    image,
    model_name
):

    x = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    benign_probability = float(
        probabilities[0].item()
    )

    malignant_probability = float(
        probabilities[1].item()
    )

    prediction = (
        "malignant"
        if malignant_probability >= 0.5
        else "benign"
    )

    return {
        "model": model_name,

        "prediction": prediction,

        "probabilities": {
            "benign": benign_probability,
            "malignant": malignant_probability
        },

        "logits": [
            float(v)
            for v in logits[0].detach().cpu().tolist()
        ]
    }


# ============================================================
# HYBRID PREDICTION
# ============================================================

def predict_hybrid(image):

    features_8d = image_to_8d(
        image
    )

    with torch.no_grad():

        logits = hybrid(
            features_8d
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

    benign_probability = float(
        probabilities[0].item()
    )

    malignant_probability = float(
        probabilities[1].item()
    )

    prediction = (
        "malignant"
        if malignant_probability >= 0.5
        else "benign"
    )

    if malignant_probability >= 0.70:

        risk_level = "High"

    elif malignant_probability >= 0.40:

        risk_level = "Moderate"

    else:

        risk_level = "Low"

    return {

        "model": (
            "EfficientNet-B3 + "
            "8-Qubit VQC"
        ),

        "prediction": prediction,

        "risk_level": risk_level,

        "probabilities": {

            "benign": benign_probability,

            "malignant": malignant_probability
        },

        "logits": [
            float(v)
            for v in logits[0].detach().cpu().tolist()
        ],

        "quantum_features": [
            float(v)
            for v in features_8d[
                0
            ].detach().cpu().numpy()
        ]
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {

        "status": "healthy",

        "project": "PROJECT QUANTUM",

        "problem": "SIH26139",

        "models": [
            "EfficientNet-B0",
            "ResNet-50",
            "EfficientNet-B3 + 8-Qubit VQC"
        ],

        "device": str(device)
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/api/models")
def models():

    return {

        "models": [

            {
                "id": "efficientnet_b0",

                "name": "EfficientNet-B0",

                "type": "Classical CNN",

                "validation_auc": (
                    b0_checkpoint.get(
                        "val_auc"
                    )
                ),

                "reported_accuracy": 0.7179,

                "reported_sensitivity": 0.7682,

                "reported_specificity": 0.6667,

                "reported_f1": 0.7332,

                "reported_roc_auc": 0.7663
            },


            {
                "id": "resnet50",

                "name": "ResNet-50",

                "type": "Classical CNN",

                "validation_auc": (
                    resnet_checkpoint.get(
                        "val_auc"
                    )
                ),

                "reported_accuracy": 0.6514,

                "reported_sensitivity": 0.6318,

                "reported_specificity": 0.6713,

                "reported_f1": 0.6465,

                "reported_roc_auc": 0.7139
            },


            {
                "id": "hybrid_vqc",

                "name": (
                    "EfficientNet-B3 + "
                    "8-Qubit VQC"
                ),

                "type": (
                    "Hybrid "
                    "Quantum-Classical"
                ),

                "qubits": 8,

                "layers": 3,

                "reported_accuracy": 0.7821,

                "reported_sensitivity": 0.7773,

                "reported_specificity": 0.7870,

                "reported_f1": 0.7826,

                "reported_roc_auc": 0.8420
            },


            {
                "id": "yolo26n",

                "name": "YOLO26n",

                "type": "Object Detection",

                "reported_precision": 0.5000,

                "reported_recall": 1.0000,

                "reported_map50": 0.5351,

                "reported_map50_95": 0.5351,

                "note": (
                    "YOLO26n checkpoint is not "
                    "loaded by this API."
                )
            }

        ]
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")


        # ----------------------------------------------------
        # Classical models
        # ----------------------------------------------------

        b0_result = predict_classical_model(
            b0,
            image,
            "EfficientNet-B0"
        )

        resnet_result = predict_classical_model(
            resnet,
            image,
            "ResNet-50"
        )


        # ----------------------------------------------------
        # Hybrid model
        # ----------------------------------------------------

        hybrid_result = predict_hybrid(
            image
        )


        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        return {

            "success": True,

            "filename": file.filename,

            "models": {

                "efficientnet_b0": b0_result,

                "resnet50": resnet_result,

                "hybrid_vqc": hybrid_result
            },

            "hybrid_result": {

                "prediction": (
                    hybrid_result[
                        "prediction"
                    ]
                ),

                "risk_level": (
                    hybrid_result[
                        "risk_level"
                    ]
                ),

                "probabilities": (
                    hybrid_result[
                        "probabilities"
                    ]
                )
            },

            "quantum_features": (
                hybrid_result[
                    "quantum_features"
                ]
            ),

            "disclaimer": (
                "Research prototype for "
                "decision support only. "
                "Not a clinical diagnosis."
            )
        }


    except Exception as e:

        return {

            "success": False,

            "error": str(e)
        }


# ============================================================
# ASYNC PREDICTION JOB
# ============================================================

def _run_prediction_job(
    job_id,
    image_bytes
):

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")


        b0_result = predict_classical_model(
            b0,
            image,
            "EfficientNet-B0"
        )


        resnet_result = predict_classical_model(
            resnet,
            image,
            "ResNet-50"
        )


        hybrid_result = predict_hybrid(
            image
        )


        result = {

            "success": True,

            "models": {

                "efficientnet_b0":
                    b0_result,

                "resnet50":
                    resnet_result,

                "hybrid_vqc":
                    hybrid_result
            },

            "disclaimer": (
                "Research prototype for "
                "decision support only. "
                "Not a clinical diagnosis."
            )
        }


        with prediction_lock:

            prediction_jobs[job_id] = {

                "status": "completed",

                "result": result
            }


    except Exception as e:

        with prediction_lock:

            prediction_jobs[job_id] = {

                "status": "failed",

                "error": str(e)
            }


# ============================================================
# ASYNC ENDPOINT
# ============================================================

@app.post("/api/predict-async")
async def predict_async(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):

    image_bytes = await file.read()

    job_id = str(
        uuid.uuid4()
    )

    with prediction_lock:

        prediction_jobs[job_id] = {

            "status": "processing"
        }

    background_tasks.add_task(
        _run_prediction_job,
        job_id,
        image_bytes
    )

    return {

        "success": True,

        "job_id": job_id,

        "status": "processing"
    }


# ============================================================
# ASYNC STATUS
# ============================================================

@app.get(
    "/api/predict-status/{job_id}"
)
async def predict_status(
    job_id: str
):

    with prediction_lock:

        job = prediction_jobs.get(
            job_id
        )

    if job is None:

        return {

            "success": False,

            "status": "not_found"
        }

    return {

        "success": True,

        "job_id": job_id,

        **job
    }