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

import gradio as gr
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
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# =========================
# MODEL CONFIG
# =========================
REPO_ID = "bartowski/Qwen2-VL-2B-Instruct-GGUF"
MODEL_FILE = "Qwen2-VL-2B-Instruct-Q5_K_M.gguf"  # Good balance: 1.13GB, high quality
MMPROJ_FILE = "mmproj-Qwen2-VL-2B-Instruct-f32.gguf"

print("Downloading model...")
model_path = hf_hub_download(REPO_ID, MODEL_FILE)
mmproj_path = hf_hub_download(REPO_ID, MMPROJ_FILE)

print("Loading model...")
llm = Llama(
    model_path=model_path,
    clip_model_path=mmproj_path,
    verbose=False,
    n_ctx=2048,  # Context length
    n_threads=8,  # Adjust based on CPU cores
    logits_all=True  # Needed for some models
)
print("Model loaded successfully")

# =========================
# PROMPTS
# =========================
SYSTEM_PROMPT = "You are a world-class AI that extracts information accurately. Follow the user's instructions precisely."
BILL_PROMPT = "Extract ALL key-value pairs from this BILL. Output ONLY the raw JSON object with no additional text, code blocks, or explanations."
INVOICE_PROMPT = "Extract ALL key-value pairs from this INVOICE. Output ONLY the raw JSON object with no additional text, code blocks, or explanations."
INSURANCE_PROMPT = "Extract ALL key-value pairs from this INSURANCE document. Output ONLY the raw JSON object with no additional text, code blocks, or explanations."

# =========================
# HELPER
# =========================
def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode('utf-8')

# =========================
# CORE INFERENCE (VISION)
# =========================
def vision_inference(base64_image: str, user_prompt: str) -> dict | str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            {"type": "text", "text": user_prompt}
        ]}
    ]

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.0
    )

    raw_output = output['choices'][0]['message']['content'].strip()

    # Clean up
    cleaned_output = re.sub(r'```json\s*|\s*```', '', raw_output).strip()

    # Try to parse as JSON and return dict if possible, else str
    try:
        return json.loads(cleaned_output)
    except json.JSONDecodeError:
        return cleaned_output

# =========================
# CORE INFERENCE (CHAT)
# =========================
def chat_inference(text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.0
    )

    return output['choices'][0]['message']['content'].strip()

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
                # Convert PIL to base64
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

# =========================
# FASTAPI APP
# =========================
app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# =========================
# APIs
# =========================

@app.get("/health")
def health():
    return {"message": "Server running !"}


# ---------- BILL ----------
@app.post("/bill")
async def bill(image: UploadFile = File(...)):
    image_bytes = await image.read()
    base64_image = image_to_base64(image_bytes)
    result = vision_inference(base64_image, BILL_PROMPT)
    return JSONResponse(content={"document": "bill", "data": result})


# ---------- INVOICE ----------
@app.post("/invoice")
async def invoice(image: UploadFile = File(...)):
    image_bytes = await image.read()
    base64_image = image_to_base64(image_bytes)
    result = vision_inference(base64_image, INVOICE_PROMPT)
    return JSONResponse(content={"document": "invoice", "data": result})


# ---------- INSURANCE ----------
@app.post("/insurance")
async def insurance(image: UploadFile = File(...)):
    image_bytes = await image.read()
    base64_image = image_to_base64(image_bytes)
    result = vision_inference(base64_image, INSURANCE_PROMPT)
    return JSONResponse(content={"document": "insurance", "data": result})


# ---------- CUSTOM (IMAGE + PROMPT FROM FRONTEND)
# RATE LIMIT: 4 requests / minute
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


# ---------- CHAT (TEXT ONLY)
# RATE LIMIT: 6 requests / minute
@app.post("/chat")
@limiter.limit("6/minute")
async def chat(
    request: Request,
    question: str = Form(...)
):
    result = chat_inference(question)
    return JSONResponse(content={"response": result})


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    # demo.launch(server_name="0.0.0.0", server_port=7860)  # Uncomment if Gradio needed
    uvicorn.run(app, host="0.0.0.0", port=8000)