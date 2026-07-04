"""A/B comparison API: base model vs fine-tuned adapter side by side."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .evaluate import base_model, finetuned_model

app = FastAPI(title="LoRA Fine-Tuning A/B Serve", version="1.0.0")
base = base_model()
tuned = finetuned_model()


class CompareRequest(BaseModel):
    clause: str


@app.post("/v1/compare")
def compare(request: CompareRequest):
    return {
        "input": request.clause,
        "base": {"model": base.name,
                 "prediction": base.predict(request.clause)},
        "finetuned": {"model": tuned.name,
                      "prediction": tuned.predict(request.clause)},
    }


@app.get("/health")
def health():
    return {"status": "ok"}
