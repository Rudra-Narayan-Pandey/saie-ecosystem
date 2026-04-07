from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/reset")
def reset():
    try:
        result = subprocess.run(
            ["python", "inference.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        return {
            "status": "ok",
            "output": result.stdout
        }
    except Exception as e:
        return {"error": str(e)}