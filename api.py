import torch
import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
import re
import base64
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
import uvicorn
from gradio import mount_gradio_app

# =========================================================
# PROMPTS
# =========================================================
BILL_PROMPT = "Extract ALL key-value pairs from this BILL. Output ONLY raw JSON."
INVOICE_PROMPT = "Extract ALL key-value pairs from this INVOICE. Output ONLY raw JSON."
INSURANCE_PROMPT = "Extract ALL key-value pairs from this INSURANCE document. Output ONLY raw JSON."

# =========================================================
# FASTAPI APP (created FIRST so health works fast)
# =========================================================
app = FastAPI(title="Document AI - Qwen2VL")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.get("/")
def root():
    return {"status": "ok", "message": "API running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# =========================================================
# MODEL LOAD (ONCE)
# =========================================================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=448 * 28 * 28
)

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model.eval()
print("Model loaded successfully")


# =========================================================
# HELPERS
# =========================================================
def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def clean_json(text: str):
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except:
        return text


# =========================================================
# CORE INFERENCE (VISION)
# =========================================================
def vision_inference(base64_image: str, prompt: str):
    image = Image.open(BytesIO(base64.b64decode(base64_image)))

    max_size = 512
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        image = image.resize(
            (int(image.width * ratio), int(image.height * ratio)),
            Image.LANCZOS
        )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = processor.decode(generated, skip_special_tokens=True)

    return clean_json(decoded)


# =========================================================
# CORE INFERENCE (CHAT)
# =========================================================
def chat_inference(text: str):
    messages = [{"role": "user", "content": text}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(text=[prompt], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)


# =========================================================
# FASTAPI ENDPOINTS
# =========================================================
@app.post("/bill")
async def bill(image: UploadFile = File(...)):
    img = await image.read()
    result = vision_inference(image_to_base64(img), BILL_PROMPT)
    return JSONResponse({"document": "bill", "data": result})


@app.post("/invoice")
async def invoice(image: UploadFile = File(...)):
    img = await image.read()
    result = vision_inference(image_to_base64(img), INVOICE_PROMPT)
    return JSONResponse({"document": "invoice", "data": result})


@app.post("/insurance")
async def insurance(image: UploadFile = File(...)):
    img = await image.read()
    result = vision_inference(image_to_base64(img), INSURANCE_PROMPT)
    return JSONResponse({"document": "insurance", "data": result})


@app.post("/custom")
@limiter.limit("4/minute")
async def custom(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    img = await image.read()
    result = vision_inference(image_to_base64(img), prompt)
    return JSONResponse({"type": "custom", "data": result})


@app.post("/chat")
@limiter.limit("6/minute")
async def chat(request: Request, question: str = Form(...)):
    return JSONResponse({"response": chat_inference(question)})


# =========================================================
# GRADIO UI
# =========================================================
with gr.Blocks(title="🧠 Multimodal Document AI") as demo:
    gr.Markdown("## 📄 Multimodal Document AI (Qwen2-VL)")

    mode = gr.Dropdown(
        ["bill", "invoice", "insurance", "custom", "chat"],
        value="bill",
        label="Mode"
    )

    image = gr.Image(type="pil", label="Upload Image")
    custom_prompt = gr.Textbox(label="Custom Prompt")
    chat_text = gr.Textbox(label="Chat Input")
    output = gr.Textbox(lines=15, label="Output")

    def ui_run(mode, image, custom_prompt, chat_text):
        if mode == "chat":
            return chat_inference(chat_text)

        if image is None:
            return "❌ Image required"

        buf = BytesIO()
        image.save(buf, format="JPEG")
        b64 = image_to_base64(buf.getvalue())

        prompt_map = {
            "bill": BILL_PROMPT,
            "invoice": INVOICE_PROMPT,
            "insurance": INSURANCE_PROMPT,
            "custom": custom_prompt
        }

        return json.dumps(
            vision_inference(b64, prompt_map[mode]),
            indent=2,
            ensure_ascii=False
        )

    gr.Button("Run 🚀").click(
        ui_run,
        inputs=[mode, image, custom_prompt, chat_text],
        outputs=output
    )


# =========================================================
# MOUNT GRADIO INSIDE FASTAPI
# =========================================================
app = mount_gradio_app(app, demo, path="/ui")


# =========================================================
# RUN SERVER
# =========================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)