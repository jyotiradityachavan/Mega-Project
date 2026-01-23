# import torch
# import gradio as gr
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from PIL import Image
# import json

# # =========================
# # MODEL CONFIG
# # =========================

# MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# print("Loading processor...")
# processor = AutoProcessor.from_pretrained(MODEL_ID)

# print("Loading model (CPU safe)...")
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.float32,   # CPU only on HF basic
#     device_map=None
# )
# model.eval()
# print("Model loaded successfully")

# # =========================
# # PROMPTS
# # =========================

# BILL_PROMPT = """
# You are a document AI system.
# Extract ALL key-value pairs from this BILL.
# Return ONLY valid JSON.
# """

# INVOICE_PROMPT = """
# You are a document AI system.
# Extract ALL key-value pairs from this INVOICE.
# Return ONLY valid JSON.
# """

# INSURANCE_PROMPT = """
# You are a document AI system.
# Extract ALL key-value pairs from this INSURANCE document.
# Return ONLY valid JSON.
# """

# # =========================
# # CORE INFERENCE
# # =========================

# def vision_infer(image: Image.Image, prompt: str) -> str:
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": prompt}
#             ]
#         }
#     ]

#     inputs = processor.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_tensors="pt"
#     )

#     with torch.no_grad():
#         outputs = model.generate(
#             inputs,
#             max_new_tokens=1024,
#             do_sample=False
#         )



#     return processor.decode(outputs[0], skip_special_tokens=True).strip()


# def chat_infer(text: str) -> str:
#     messages = [{"role": "user", "content": text}]

#     inputs = processor.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_tensors="pt"
#     )

#     with torch.no_grad():
#         outputs = model.generate(
#             inputs,
#             max_new_tokens=512,
#             do_sample=False
#         )


#     return processor.decode(outputs[0], skip_special_tokens=True).strip()

# # =========================
# # ROUTER
# # =========================

# def run_api(endpoint, image, custom_prompt, chat_text):
#     try:
#         if endpoint in ["bill", "invoice", "insurance"]:
#             if image is None:
#                 return "❌ Image required"

#             prompt_map = {
#                 "bill": BILL_PROMPT,
#                 "invoice": INVOICE_PROMPT,
#                 "insurance": INSURANCE_PROMPT
#             }

#             result = vision_infer(image, prompt_map[endpoint])
#             return json.dumps(
#                 {"document": endpoint, "data": result},
#                 indent=2,
#                 ensure_ascii=False
#             )

#         elif endpoint == "custom":
#             if image is None or not custom_prompt:
#                 return "❌ Image + custom prompt required"

#             result = vision_infer(image, custom_prompt)
#             return json.dumps(
#                 {"type": "custom", "data": result},
#                 indent=2,
#                 ensure_ascii=False
#             )

#         elif endpoint == "chat":
#             if not chat_text:
#                 return "❌ Chat text required"

#             result = chat_infer(chat_text)
#             return json.dumps(
#                 {"response": result},
#                 indent=2,
#                 ensure_ascii=False
#             )

#         else:
#             return "❌ Invalid mode selected"

#     except Exception as e:
#         return f"❌ Error: {str(e)}"

# # =========================
# # GRADIO UI (SAFE MODE)
# # =========================

# with gr.Blocks(
#     title="🧠 Multimodal Document AI (Qwen2-VL)",
#     analytics_enabled=False
# ) as demo:

#     gr.Markdown("## 📄 Multimodal Document AI (Qwen2-VL Vision)")

#     endpoint = gr.Dropdown(
#         ["bill", "invoice", "insurance", "custom", "chat"],
#         label="Select Mode"
#     )

#     image = gr.Image(type="pil", label="Upload Document Image")
#     custom_prompt = gr.Textbox(label="Custom Prompt (custom mode)")
#     chat_text = gr.Textbox(label="Chat Text (chat mode)")
#     output = gr.Textbox(lines=18, label="Response (JSON)")

#     run_btn = gr.Button("Run Inference 🚀")

#     # 🔴 CRITICAL FIX: Disable Gradio API schema
#     run_btn.click(
#         fn=run_api,
#         inputs=[endpoint, image, custom_prompt, chat_text],
#         outputs=output,
#         api_name=False
#     )

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860)

import torch
import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json, re, base64
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import uvicorn

# =========================
# MODEL CONFIG (CPU SAFE)
# =========================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=224 * 224,
    max_pixels=384 * 384
)

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)
model.eval()
print("Model loaded successfully")

# =========================
# PROMPTS
# =========================
BILL_PROMPT = "Extract ALL key-value pairs from this BILL. Output ONLY raw JSON."
INVOICE_PROMPT = "Extract ALL key-value pairs from this INVOICE. Output ONLY raw JSON."
INSURANCE_PROMPT = "Extract ALL key-value pairs from this INSURANCE document. Output ONLY raw JSON."

# =========================
# HELPERS
# =========================
def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def vision_inference(base64_image: str, prompt: str):
    image_bytes = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    if max(image.size) > 512:
        ratio = 512 / max(image.size)
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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text_out = processor.decode(generated, skip_special_tokens=True)

    text_out = re.sub(r"```json|```", "", text_out).strip()

    try:
        return json.loads(text_out)
    except:
        return text_out

def chat_inference(text: str):
    messages = [{"role": "user", "content": text}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)

# =========================
# FASTAPI SETUP
# =========================
app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return PlainTextResponse("Rate limit exceeded", status_code=429)

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {"status": "API running"}

# =========================
# APIs
# =========================
@app.post("/bill")
async def bill(image: UploadFile = File(...)):
    data = await image.read()
    return {"document": "bill", "data": vision_inference(image_to_base64(data), BILL_PROMPT)}

@app.post("/invoice")
async def invoice(image: UploadFile = File(...)):
    data = await image.read()
    return {"document": "invoice", "data": vision_inference(image_to_base64(data), INVOICE_PROMPT)}

@app.post("/insurance")
async def insurance(image: UploadFile = File(...)):
    data = await image.read()
    return {"document": "insurance", "data": vision_inference(image_to_base64(data), INSURANCE_PROMPT)}

@app.post("/custom")
@limiter.limit("4/minute")
async def custom(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    data = await image.read()
    return {"type": "custom", "data": vision_inference(image_to_base64(data), prompt)}

@app.post("/chat")
@limiter.limit("6/minute")
async def chat(
    request: Request,
    question: str = Form(...)
):
    return {"response": chat_inference(question)}

# =========================
# RUN
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
