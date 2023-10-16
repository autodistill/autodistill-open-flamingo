import os
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

torch.use_deterministic_algorithms(False)

import numpy as np
import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class OpenFlamingo(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology, confidence: int = 0.5):
        self.ontology = ontology

        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
        )

        self.open_flamingo_model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        checkpoint_path = hf_hub_download(
            "openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt"
        )
        model.load_state_dict(torch.load(checkpoint_path), strict=False)

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        pass
