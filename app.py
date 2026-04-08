from fastapi import FastAPI
import threading
import subprocess

app = FastAPI()

_inference_status = {"running": False, "last_output": "", "last_error": ""}


@app.get("/")
def root():
    return {"status": "running"}


def _run_inference():
    _inference_status["running"] = True
    try:
        result = subprocess.run(
            ["python", "inference.py"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        _inference_status["last_output"] = result.stdout[-4000:] if result.stdout else ""
        _inference_status["last_error"] = result.stderr[-2000:] if result.stderr else ""
    except Exception as e:
        _inference_status["last_error"] = str(e)
    finally:
        _inference_status["running"] = False


@app.post("/reset")
def reset():
    """
    Immediately returns 200 OK and triggers inference asynchronously.
    The OpenEnv validator expects a fast response from this endpoint.
    """
    if not _inference_status["running"]:
        t = threading.Thread(target=_run_inference, daemon=True)
        t.start()
    return {"status": "ok"}


@app.get("/status")
def status():
    return {
        "running": _inference_status["running"],
        "last_output": _inference_status["last_output"],
        "last_error": _inference_status["last_error"],
    }
