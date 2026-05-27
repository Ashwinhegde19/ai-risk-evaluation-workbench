"""Model client factories."""

from __future__ import annotations

import os

from models.modal_endpoint import ModalEndpointClient
from models.oss_hf import HuggingFaceOSSClient


def create_oss_client():
    """Create the OSS assistant backend from environment configuration."""

    backend = os.getenv("OSS_BACKEND", "local").strip().lower()
    if backend in {"local", "hf", "huggingface"}:
        return HuggingFaceOSSClient()
    if backend == "modal":
        return ModalEndpointClient()
    raise ValueError(f"Unsupported OSS_BACKEND: {backend}")
