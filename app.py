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
from qwen_vl_utils import process_vision_info  # Critical import for vision processing
from PIL import Image
import json
import re  # For cleaning up the model's output

# =========================
# MODEL CONFIG
# =========================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model (CPU safe)...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,           # Use float32 for CPU compatibility
    device_map="cpu",                    # Explicit CPU; change to "auto" or "cuda" if GPU available in your HF Space
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model.eval()
print("Model loaded successfully")

# =========================
# PROMPTS (Stricter to avoid markdown wrappers)
# =========================
BILL_PROMPT = "Extract ALL key-value pairs from this BILL. Output ONLY the raw JSON object with no additional text, code blocks, or explanations."
INVOICE_PROMPT = "Extract ALL key-value pairs from this INVOICE. Output ONLY the raw JSON object with no additional text, code blocks, or explanations."
INSURANCE_PROMPT = "Extract ALL key-value pairs from this INSURANCE document. Output ONLY the raw JSON object with no additional text, code blocks, or explanations."

# =========================
# CORE INFERENCE (VISION) - Reduced max_new_tokens for speed
# =========================
def vision_infer(image: Image.Image, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # PIL Image directly
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Apply chat template (adds generation prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process vision info (critical step!)
    image_inputs, video_inputs = process_vision_info(messages)

    # Full processing
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # Reduced from 1024 for faster inference
            do_sample=False
        )

    generated_ids_trimmed = outputs[0][inputs['input_ids'].shape[1]:]  # Trim input part
    raw_output = processor.decode(generated_ids_trimmed, skip_special_tokens=True).strip()

    # Clean up any unwanted wrappers (e.g., ```json ... ```)
    cleaned_output = re.sub(r'```json\s*|\s*```', '', raw_output).strip()
    return cleaned_output

# =========================
# CORE INFERENCE (CHAT - TEXT ONLY) - Reduced max_new_tokens for speed
# =========================
def chat_infer(text: str) -> str:
    messages = [{"role": "user", "content": text}]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt],
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Reduced from 512 for faster inference
            do_sample=False
        )

    generated_ids_trimmed = outputs[0][inputs['input_ids'].shape[1]:]
    return processor.decode(generated_ids_trimmed, skip_special_tokens=True).strip()

# =========================
# ROUTER
# =========================
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
            result = vision_infer(image, prompt_map[endpoint])
            # Parse and validate as JSON
            try:
                json_data = json.loads(result)
                return json.dumps({"document": endpoint, "data": json_data}, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                return json.dumps({"document": endpoint, "data": result}, indent=2, ensure_ascii=False)  # Fallback to string if not valid JSON

        elif endpoint == "custom":
            if image is None or not custom_prompt:
                return "❌ Image + custom prompt required"
            result = vision_infer(image, custom_prompt)
            return json.dumps({"type": "custom", "data": result}, indent=2, ensure_ascii=False)

        elif endpoint == "chat":
            if not chat_text:
                return "❌ Chat text required"
            result = chat_infer(chat_text)
            return json.dumps({"response": result}, indent=2, ensure_ascii=False)

        else:
            return "❌ Invalid mode selected"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# =========================
# GRADIO UI
# =========================
with gr.Blocks(
    title="🧠 Multimodal Document AI (Qwen2-VL)",
    analytics_enabled=False
) as demo:
    gr.Markdown("## 📄 Multimodal Document AI (Qwen2-VL Vision)")
    endpoint = gr.Dropdown(
        ["bill", "invoice", "insurance", "custom", "chat"],
        label="Select Mode",
        value="bill"
    )
    image = gr.Image(type="pil", label="Upload Document Image")
    custom_prompt = gr.Textbox(label="Custom Prompt (custom mode)")
    chat_text = gr.Textbox(label="Chat Text (chat mode)")
    output = gr.Textbox(lines=18, label="Response (JSON)")
    run_btn = gr.Button("Run Inference 🚀")

    run_btn.click(
        fn=run_api,
        inputs=[endpoint, image, custom_prompt, chat_text],
        outputs=output,
        api_name=False
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)