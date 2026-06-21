```python
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import torch
import re

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)

model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()

def summarize_dialogue(dialogue):
    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}




@app.get("/health")
async def health():
    return {"status": "healthy"}