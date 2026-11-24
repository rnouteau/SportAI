"""SportAI API: read-only access to the analyses written by the pipeline.

    uvicorn api.main:app --reload        # then http://localhost:8000/docs

``SPORTAI_OUTPUTS`` sets the folder read (default ``outputs/tennis``).
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.analyses import FILES_URL, list_analyses, load_summary
from api.schemas import AnalysisSummary

FRONTEND_ORIGINS = ["http://localhost:5173"]  # Vite dev server


def create_app(outputs_root: Path) -> FastAPI:
    """The API reading ``outputs_root`` (tests give it a temporary folder)."""
    app = FastAPI(title="SportAI", description="Tennis video analyses")
    app.add_middleware(CORSMiddleware, allow_origins=FRONTEND_ORIGINS, allow_methods=["GET"])

    @app.get("/api/analyses", response_model=list[AnalysisSummary])
    def get_analyses() -> list[AnalysisSummary]:
        return list_analyses(outputs_root)

    @app.get("/api/analyses/{analysis_id}", response_model=AnalysisSummary)
    def get_analysis(analysis_id: str) -> AnalysisSummary:
        summary = load_summary(outputs_root, analysis_id)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"Analysis {analysis_id!r} not found")
        return summary

    # Videos and images (URLs given in the summaries). Mounted last: the routes above win.
    app.mount(FILES_URL, StaticFiles(directory=outputs_root, check_dir=False), name="files")
    return app


app = create_app(Path(os.environ.get("SPORTAI_OUTPUTS", "outputs/tennis")))
