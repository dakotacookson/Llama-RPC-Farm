"""Model file replication between machines (settings-gated)."""
import pydantic
from fastapi import APIRouter, HTTPException

import model_sync

router = APIRouter()


class EnsureIn(pydantic.BaseModel):
    model_id: str | None = None


@router.get("/api/model-sync/status")
async def model_sync_status():
    """Settings + the newest background sync job (if any)."""
    latest = model_sync.latest_job()
    return {"config": model_sync.sync_config(), "job": latest}


@router.post("/api/model-sync/ensure")
async def model_sync_ensure(body: EnsureIn | None = None):
    model_id = (body.model_id.strip() if body and body.model_id else None)
    if model_id:
        import config_loader as cfg
        if cfg.get_model(model_id) is None:
            raise HTTPException(404, f"model '{model_id}' not found")
    res = model_sync.start_job(model_id)
    if not res.get("ok"):
        raise HTTPException(400, res.get("error") or "could not start sync")
    return res


@router.get("/api/model-sync/jobs/{job_id}")
async def model_sync_job(job_id: str):
    st = model_sync.job_status(job_id)
    if not st:
        raise HTTPException(404, "unknown sync job")
    return st
