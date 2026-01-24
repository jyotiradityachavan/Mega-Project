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


# import torch
# import gradio as gr
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from PIL import Image
# import json
# import re
# from fastapi import FastAPI, UploadFile, File, Form, Request
# from fastapi.responses import JSONResponse
# from slowapi import Limiter
# from slowapi.util import get_remote_address
# import uvicorn
# import base64
# from io import BytesIO

# # =========================
# # MODEL CONFIG
# # =========================
# MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# print("Loading processor...")
# processor = AutoProcessor.from_pretrained(
#     MODEL_ID,
#     min_pixels=256 * 28 * 28,          # Keep min reasonable
#     max_pixels=448 * 28 * 28           # Lower max (~350k pixels) for faster vision processing
# )

# print("Loading model...")
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     torch_dtype="auto",                # Auto-select best (bf16/f32)
#     device_map="auto",                 # Use GPU if available!
#     low_cpu_mem_usage=True,
#     trust_remote_code=True
# )
# model.eval()
# print("Model loaded successfully")

# # =========================
# # PROMPTS
# # =========================
# BILL_PROMPT = "Extract ALL key-value pairs from this BILL. Output ONLY the raw JSON object with no additional text, code blocks, or explanations."
# INVOICE_PROMPT = "Extract ALL key-value pairs from this INVOICE. Output ONLY the raw JSON object with no additional text, code blocks, or explanations."
# INSURANCE_PROMPT = "Extract ALL key-value pairs from this INSURANCE document. Output ONLY the raw JSON object with no additional text, code blocks, or explanations."

# # =========================
# # HELPER
# # =========================
# def image_to_base64(image_bytes: bytes) -> str:
#     return base64.b64encode(image_bytes).decode('utf-8')

# # =========================
# # CORE INFERENCE (VISION)
# # =========================
# def vision_inference(base64_image: str, prompt: str) -> str:
#     # Decode base64 to bytes
#     image_bytes = base64.b64decode(base64_image)
#     image = Image.open(BytesIO(image_bytes))

#     # Resize image for speed (preserve aspect ratio)
#     max_size = 512
#     if image.width > max_size or image.height > max_size:
#         ratio = max_size / max(image.width, image.height)
#         new_size = (int(image.width * ratio), int(image.height * ratio))
#         image = image.resize(new_size, Image.LANCZOS)

#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": prompt}
#             ]
#         }
#     ]

#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     image_inputs, video_inputs = process_vision_info(messages)

#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt"
#     )

#     inputs = {k: v.to(model.device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=256,  # Reduced for speed
#             do_sample=False
#         )

#     generated_ids_trimmed = outputs[0][inputs['input_ids'].shape[1]:]
#     raw_output = processor.decode(generated_ids_trimmed, skip_special_tokens=True).strip()

#     # Clean up
#     cleaned_output = re.sub(r'```json\s*|\s*```', '', raw_output).strip()

#     # Try to parse as JSON and return dict if possible, else str
#     try:
#         return json.loads(cleaned_output)
#     except json.JSONDecodeError:
#         return cleaned_output

# # =========================
# # CORE INFERENCE (CHAT)
# # =========================
# def chat_inference(text: str) -> str:
#     messages = [{"role": "user", "content": text}]
#     text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = processor(text=[text_prompt], return_tensors="pt")
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=256,
#             do_sample=False
#         )

#     generated_ids_trimmed = outputs[0][inputs['input_ids'].shape[1]:]
#     return processor.decode(generated_ids_trimmed, skip_special_tokens=True).strip()

# # =========================
# # GRADIO UI (Kept as is)
# # =========================
# with gr.Blocks(title="🧠 Multimodal Document AI (Qwen2-VL)") as demo:
#     gr.Markdown("## 📄 Multimodal Document AI (Qwen2-VL Vision)")
#     endpoint = gr.Dropdown(["bill", "invoice", "insurance", "custom", "chat"], label="Select Mode", value="bill")
#     image = gr.Image(type="pil", label="Upload Document Image")
#     custom_prompt = gr.Textbox(label="Custom Prompt (custom mode)")
#     chat_text = gr.Textbox(label="Chat Text (chat mode)")
#     output = gr.Textbox(lines=18, label="Response (JSON)")
#     run_btn = gr.Button("Run Inference 🚀")

#     def run_api(endpoint, image, custom_prompt, chat_text):
#         try:
#             if endpoint in ["bill", "invoice", "insurance"]:
#                 if image is None:
#                     return "❌ Image required"
#                 prompt_map = {
#                     "bill": BILL_PROMPT,
#                     "invoice": INVOICE_PROMPT,
#                     "insurance": INSURANCE_PROMPT
#                 }
#                 # For Gradio, image is PIL.Image, so convert to bytes
#                 buffered = BytesIO()
#                 image.save(buffered, format="JPEG")  # or PNG if needed
#                 image_bytes = buffered.getvalue()
#                 base64_image = image_to_base64(image_bytes)
#                 result = vision_inference(base64_image, prompt_map[endpoint])
#                 return json.dumps({"document": endpoint, "data": result}, indent=2, ensure_ascii=False)

#             elif endpoint == "custom":
#                 if image is None or not custom_prompt:
#                     return "❌ Image + custom prompt required"
#                 buffered = BytesIO()
#                 image.save(buffered, format="JPEG")
#                 image_bytes = buffered.getvalue()
#                 base64_image = image_to_base64(image_bytes)
#                 result = vision_inference(base64_image, custom_prompt)
#                 return json.dumps({"type": "custom", "data": result}, indent=2, ensure_ascii=False)

#             elif endpoint == "chat":
#                 if not chat_text:
#                     return "❌ Chat text required"
#                 result = chat_inference(chat_text)
#                 return json.dumps({"response": result}, indent=2, ensure_ascii=False)

#             else:
#                 return "❌ Invalid mode selected"

#         except Exception as e:
#             return f"❌ Error: {str(e)}"

#     run_btn.click(fn=run_api, inputs=[endpoint, image, custom_prompt, chat_text], outputs=output)

# # =========================
# # FASTAPI APP
# # =========================
# app = FastAPI()
# limiter = Limiter(key_func=get_remote_address)
# app.state.limiter = limiter

# # =========================
# # APIs
# # =========================

# @app.get("/health")
# def health():
#     return {"message": "Server running !"}


# # ---------- BILL ----------
# @app.post("/bill")
# async def bill(image: UploadFile = File(...)):
#     image_bytes = await image.read()
#     base64_image = image_to_base64(image_bytes)
#     result = vision_inference(base64_image, BILL_PROMPT)
#     return JSONResponse(content={"document": "bill", "data": result})


# # ---------- INVOICE ----------
# @app.post("/invoice")
# async def invoice(image: UploadFile = File(...)):
#     image_bytes = await image.read()
#     base64_image = image_to_base64(image_bytes)
#     result = vision_inference(base64_image, INVOICE_PROMPT)
#     return JSONResponse(content={"document": "invoice", "data": result})


# # ---------- INSURANCE ----------
# @app.post("/insurance")
# async def insurance(image: UploadFile = File(...)):
#     image_bytes = await image.read()
#     base64_image = image_to_base64(image_bytes)
#     result = vision_inference(base64_image, INSURANCE_PROMPT)
#     return JSONResponse(content={"document": "insurance", "data": result})


# # ---------- CUSTOM (IMAGE + PROMPT FROM FRONTEND)
# # RATE LIMIT: 4 requests / minute
# @app.post("/custom")
# @limiter.limit("4/minute")
# async def custom(
#     request: Request,
#     image: UploadFile = File(...),
#     prompt: str = Form(...)
# ):
#     image_bytes = await image.read()
#     base64_image = image_to_base64(image_bytes)
#     result = vision_inference(base64_image, prompt)
#     return JSONResponse(content={"type": "custom", "data": result})


# # ---------- CHAT (TEXT ONLY)
# # RATE LIMIT: 6 requests / minute
# @app.post("/chat")
# @limiter.limit("6/minute")
# async def chat(
#     request: Request,
#     question: str = Form(...)
# ):
#     result = chat_inference(question)
#     return JSONResponse(content={"response": result})


# # =========================
# # RUN SERVER
# # =========================
# if __name__ == "__main__":
#     # Launch Gradio on 7860 (if desired, but for APIs, focus on FastAPI)
#     # demo.launch(server_name="0.0.0.0", server_port=7860)  # Comment out if not needed
#     uvicorn.run(app, host="0.0.0.0", port=8000)



import torch
import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
import re
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
import uvicorn
import base64
from io import BytesIO
from gradio import mount_gradio_app

# =========================
# PROMPTS (defined early)
# =========================
BILL_PROMPT = "Extract ALL key-value pairs from this BILL. Output ONLY the raw JSON object with no additional text, code blocks, or explanations."
INVOICE_PROMPT = "Extract ALL key-value pairs from this INVOICE. Output ONLY the raw JSON object with no additional text, code blocks, or explanations."
INSURANCE_PROMPT = "Extract ALL key-value pairs from this INSURANCE document. Output ONLY the raw JSON object with no additional text, code blocks, or explanations."

# =========================
# FASTAPI APP (defined early so health endpoints respond fast)
# =========================
app = FastAPI(title="Document AI - Qwen2-VL")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/ready")  # HF sometimes probes /ready or /live
async def ready():
    return {"ready": True}
    
# Fast root endpoint for HF health probes
@app.get("/")
async def root():
    return {"status": "ok", "message": "API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "message": "Server running !"}

# =========================
# MODEL & PROCESSOR (loaded AFTER routes - so health responds during loading)
# =========================
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

# =========================
# HELPER
# =========================
def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')

# =========================
# CORE INFERENCE (VISION)
# =========================
def vision_inference(base64_image: str, prompt: str):
    image_bytes = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_bytes))

    max_size = 512
    if image.width > max_size or image.height > max_size:
        ratio = max_size / max(image.width, image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

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

    generated_ids_trimmed = outputs[0][inputs['input_ids'].shape[1]:]
    raw_output = processor.decode(generated_ids_trimmed, skip_special_tokens=True).strip()
    cleaned_output = re.sub(r'```json\s*|\s*```', '', raw_output).strip()

    try:
        return json.loads(cleaned_output)
    except json.JSONDecodeError:
        return cleaned_output

# =========================
# CORE INFERENCE (CHAT)
# =========================
def chat_inference(text: str) -> str:
    messages = [{"role": "user", "content": text}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    generated_ids_trimmed = outputs[0][inputs['input_ids'].shape[1]:]
    return processor.decode(generated_ids_trimmed, skip_special_tokens=True).strip()

# =========================
# FASTAPI ENDPOINTS (document extraction)
# =========================
@app.post("/bill")
async def bill(image: UploadFile = File(...)):
    image_bytes = await image.read()
    base64_image = image_to_base64(image_bytes)
    result = vision_inference(base64_image, BILL_PROMPT)
    return JSONResponse(content={"document": "bill", "data": result})

@app.post("/invoice")
async def invoice(image: UploadFile = File(...)):
    image_bytes = await image.read()
    base64_image = image_to_base64(image_bytes)
    result = vision_inference(base64_image, INVOICE_PROMPT)
    return JSONResponse(content={"document": "invoice", "data": result})

@app.post("/insurance")
async def insurance(image: UploadFile = File(...)):
    image_bytes = await image.read()
    base64_image = image_to_base64(image_bytes)
    result = vision_inference(base64_image, INSURANCE_PROMPT)
    return JSONResponse(content={"document": "insurance", "data": result})

@app.post("/custom")
@limiter.limit("4/minute")
async def custom(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    image_bytes = await image.read()
    base64_image = image_to_base64(image_bytes)
    result = vision_inference(base64_image, prompt)
    return JSONResponse(content={"type": "custom", "data": result})

@app.post("/chat")
@limiter.limit("6/minute")
async def chat(
    request: Request,
    question: str = Form(...)
):
    result = chat_inference(question)
    return JSONResponse(content={"response": result})

# =========================
# GRADIO UI
# =========================
with gr.Blocks(title="🧠 Multimodal Document AI (Qwen2-VL)") as demo:
    gr.Markdown("## 📄 Multimodal Document AI (Qwen2-VL Vision)")
    endpoint = gr.Dropdown(["bill", "invoice", "insurance", "custom", "chat"], label="Select Mode", value="bill")
    image = gr.Image(type="pil", label="Upload Document Image")
    custom_prompt = gr.Textbox(label="Custom Prompt (custom mode)")
    chat_text = gr.Textbox(label="Chat Text (chat mode)")
    output = gr.Textbox(lines=18, label="Response (JSON)")
    run_btn = gr.Button("Run Inference 🚀")

    def run_api(endpoint, image, custom_prompt, chat_text):
        try:
            if endpoint in ["bill", "invoice", "insurance"]:
                if image is None:
                    return "❌ Image required"
                prompt_map = {
                    "bill": BILL_PROMPT,
                    "invoice": INVOICE_PROMPT,
                    "insurance": INSURANCE_PROMPT
                }
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                image_bytes = buffered.getvalue()
                base64_image = image_to_base64(image_bytes)
                result = vision_inference(base64_image, prompt_map[endpoint])
                return json.dumps({"document": endpoint, "data": result}, indent=2, ensure_ascii=False)

            elif endpoint == "custom":
                if image is None or not custom_prompt:
                    return "❌ Image + custom prompt required"
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                image_bytes = buffered.getvalue()
                base64_image = image_to_base64(image_bytes)
                result = vision_inference(base64_image, custom_prompt)
                return json.dumps({"type": "custom", "data": result}, indent=2, ensure_ascii=False)

            elif endpoint == "chat":
                if not chat_text:
                    return "❌ Chat text required"
                result = chat_inference(chat_text)
                return json.dumps({"response": result}, indent=2, ensure_ascii=False)

            else:
                return "❌ Invalid mode selected"

        except Exception as e:
            return f"❌ Error: {str(e)}"

    run_btn.click(fn=run_api, inputs=[endpoint, image, custom_prompt, chat_text], outputs=output)

# Mount Gradio inside FastAPI
app = mount_gradio_app(app, demo, path="/ui")

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    try:
        print("Starting Uvicorn server...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        print(f"Uvicorn server failed: {str(e)}")
        raise
