from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import librosa
from helper import transcribe_audio

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio, sr = librosa.load(file.file, sr=None)
    transcription = transcribe_audio(audio)
    return JSONResponse(content={"transcription": transcription})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
