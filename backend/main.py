"""
NEXTMOVE V4 — Decision Visualization Engine.

One backend service. One LLM call. One deterministic engine.
No scores. No ranking. No recommendation. Only structure.
"""
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from llm_client import extract_job_analysis
from engine import build_decision_card
from models import AnalyzeRequest, DecisionCard, ShareCreateRequest, ShareCreateResponse
import questionnaire
import storage

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
LANDING_DIR = BASE_DIR / "landing"

NEXTMOVE_USER = os.environ.get("NEXTMOVE_USER", "olivier")
NEXTMOVE_PASSWORD = os.environ.get("NEXTMOVE_PASSWORD", "olivier_nextmove")

app = FastAPI(title="NEXTMOVE V4", description="Decision Visualization Engine")
app.mount("/landing/images", StaticFiles(directory=LANDING_DIR / "images"), name="landing-images")
security = HTTPBasic()


def require_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    user_ok = secrets.compare_digest(credentials.username, NEXTMOVE_USER)
    password_ok = secrets.compare_digest(credentials.password, NEXTMOVE_PASSWORD)
    if not (user_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.post("/analyze", response_model=DecisionCard)
def analyze(request: AnalyzeRequest, _user: str = Depends(require_auth)) -> DecisionCard:
    try:
        extraction, repro_hash = extract_job_analysis(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM extraction failed: {exc}") from exc

    return build_decision_card(extraction, request.profile, reproducibility_hash=repro_hash)


@app.get("/questionnaire")
def get_questionnaire():
    return {"questions": questionnaire.get_ordered_questions()}


@app.post("/questionnaire/score")
def score_questionnaire(answers: dict[str, str]):
    return questionnaire.compute_profile_vector(answers)


@app.post("/share", response_model=ShareCreateResponse)
def create_share(request: ShareCreateRequest, _user: str = Depends(require_auth)) -> ShareCreateResponse:
    share_id = storage.create_share(
        current_job_title=request.current_job_title,
        opportunity_title=request.opportunity_title,
        payload=request.result.model_dump(),
    )
    return ShareCreateResponse(share_id=share_id, url=f"/share/{share_id}")


@app.get("/")
def serve_landing() -> FileResponse:
    return FileResponse(LANDING_DIR / "index.html")


@app.get("/quiz")
def serve_quiz() -> FileResponse:
    return FileResponse(LANDING_DIR / "quiz.html")


@app.get("/app")
def serve_app() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
