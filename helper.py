import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

def load_model():
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        model.to(device)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(f"Error loading processor: {e}")

    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
    except Exception as e:
        raise RuntimeError(f"Error initializing pipeline: {e}")
    
    return pipe

pipe = load_model()

def transcribe_audio(audio):
    try:
        # Ensure the audio is in the correct format (list of floats)
        if not isinstance(audio, list):
            audio = audio.tolist()

        result = pipe(audio)
        return result["text"]
    except Exception as e:
        raise RuntimeError(f"Error during transcription: {e}")
