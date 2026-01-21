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
import json

# =========================
# GLOBAL SAFETY SETTINGS
# =========================
torch.set_grad_enabled(False)

# =========================
# MODEL CONFIG
# =========================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model (CPU SAFE)...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,      # ✅ CPU SAFE
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model.eval()
print("Model loaded successfully")

# =========================
# PROMPTS
# =========================
BILL_PROMPT = "Extract ALL key-value pairs from this BILL. Return ONLY valid JSON."
INVOICE_PROMPT = "Extract ALL key-value pairs from this INVOICE. Return ONLY valid JSON."
INSURANCE_PROMPT = "Extract ALL key-value pairs from this INSURANCE document. Return ONLY valid JSON."

# =========================
# VISION INFERENCE
# =========================
def vision_infer(image: Image.Image, prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,     # ✅ SAFE LIMIT
        do_sample=False
    )

    generated = outputs[0][inputs.input_ids.shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()

# =========================
# CHAT (TEXT ONLY)
# =========================
def chat_infer(text: str) -> str:
    messages = [{"role": "user", "content": text}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False
    )

    generated = outputs[0][inputs.input_ids.shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()

# =========================
# ROUTER
# =========================
def run_api(mode, image, custom_prompt, chat_text):
    try:
        if mode in ["bill", "invoice", "insurance"]:
            if image is None:
                return "❌ Image required"

            prompt_map = {
                "bill": BILL_PROMPT,
                "invoice": INVOICE_PROMPT,
                "insurance": INSURANCE_PROMPT
            }

            result = vision_infer(image, prompt_map[mode])
            return json.dumps(
                {"document_type": mode, "raw_output": result},
                indent=2,
                ensure_ascii=False
            )

        elif mode == "custom":
            if image is None or not custom_prompt:
                return "❌ Image + custom prompt required"

            result = vision_infer(image, custom_prompt)
            return json.dumps(
                {"document_type": "custom", "raw_output": result},
                indent=2,
                ensure_ascii=False
            )

        elif mode == "chat":
            if not chat_text:
                return "❌ Chat text required"

            result = chat_infer(chat_text)
            return json.dumps(
                {"response": result},
                indent=2,
                ensure_ascii=False
            )

        return "❌ Invalid mode"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# =========================
# GRADIO UI
# =========================
with gr.Blocks(
    title="🧠 Multimodal Document AI (Qwen2-VL)",
    analytics_enabled=False
) as demo:

    gr.Markdown("## 📄 Multimodal Document AI (Qwen2-VL-2B)")

    mode = gr.Dropdown(
        ["bill", "invoice", "insurance", "custom", "chat"],
        value="bill",
        label="Select Mode"
    )

    image = gr.Image(type="pil", label="Upload Document Image")
    custom_prompt = gr.Textbox(label="Custom Prompt (custom mode)")
    chat_text = gr.Textbox(label="Chat Text (chat mode)")

    output = gr.Textbox(lines=18, label="Response (JSON)")
    run_btn = gr.Button("Run Inference 🚀")

    run_btn.click(
        fn=run_api,
        inputs=[mode, image, custom_prompt, chat_text],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
