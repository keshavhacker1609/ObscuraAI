"""Upload router — kept minimal, used for future CSV embedding uploads."""
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post("")
@router.post("/")
async def upload_embeddings(file: UploadFile = File(...)):
    """
    Upload a custom .npz embedding file for auditing.
    (Future: replace synthetic data with your own extracted embeddings)
    """
    if not file.filename.endswith(".npz"):
        raise HTTPException(400, "Only .npz files supported.")

    import tempfile, shutil
    from pathlib import Path
    from ml_engine.config import DATA_DIR

    dest = DATA_DIR / "uploaded_embeddings.npz"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "status":   "uploaded",
        "filename": file.filename,
        "saved_to": str(dest),
        "message":  "File uploaded. Use ?dataset=uploaded in pipeline run to use this file.",
    }
