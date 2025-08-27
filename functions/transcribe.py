### Transcribe

import streamlit as st
import torch
from datasets import load_dataset
from transformers import pipeline
import whisper

# Functions that transcribes audio and creates the text files

def transcribe_with_kb_whisper(file_name_converted, file_name, whisper_model, spoken_language):
	
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

	pipe = pipeline(
		"automatic-speech-recognition",
		model=f"KBLab/{whisper_model}",
		torch_dtype=torch_dtype,
		device=device,
	)

	res = pipe(
        file_name_converted,
        chunk_length_s=30,
        ignore_warning=True, # Lägger till denna parameter för att ignorera varningen
        generate_kwargs={"language": spoken_language, "task": "transcribe"}
    )
	
	transcribed_content = res["text"]

	with open('text/' + file_name + '.txt', 'w', encoding='utf-8', errors='replace') as file:
		# Write the string to the file
		file.write(transcribed_content)
	
	return transcribed_content


def transcribe_with_whisper(file_name_converted, file_name, whisper_model, spoken_language):

	transcribed_content = ""
	
	# Whisper models are now directly loaded from the openai-whisper package
	model = whisper.load_model(whisper_model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
	result = model.transcribe(file_name_converted, language=spoken_language)
	transcribed_content = result["text"]

	with open('text/' + file_name + '.txt', 'w', encoding='utf-8', errors='replace') as file:
		# Write the string to the file
		file.write(transcribed_content)

	return transcribed_content