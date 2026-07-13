from fastapi import FastAPI

from gmoney.settings import get_settings

app = FastAPI(title="GMoney V2 API", version="0.1.0", docs_url="/v2/docs")


@app.get("/health/live", tags=["health"])
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready", tags=["health"])
def ready() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ready",
        "environment": settings.env,
        "tenant_mode": "fixed-development",
    }

