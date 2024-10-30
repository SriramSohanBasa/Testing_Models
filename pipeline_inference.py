import warnings
from transformers import pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

transcriber = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-large-v2",
    device="mps",
    model_kwargs={language : "en"}  # Explicitly set language to English
)

# Perform transcription on the provided audio URL
result = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(result)