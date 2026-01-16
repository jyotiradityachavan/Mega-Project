# import gradio as gr
# import torch
# import json
# from PIL import Image
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import json


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

print("Loading Qwen2-VL processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading Qwen2-VL model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("Model loaded successfully!")

# =========================
# PROMPTS
# =========================

BILL_PROMPT = """
You are a document AI system.
Extract ALL key-value pairs from this BILL.
Return ONLY valid JSON.
"""

INVOICE_PROMPT = """
You are a document AI system.
Extract ALL key-value pairs from this INVOICE.
Return ONLY valid JSON.
"""

INSURANCE_PROMPT = """
You are a document AI system.
Extract ALL key-value pairs from this INSURANCE document.
Return ONLY valid JSON.
"""

# =========================
# CORE INFERENCE
# =========================

def vision_infer(image: Image.Image, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=1024,
        temperature=0.0
    )

    result = processor.decode(outputs[0], skip_special_tokens=True)
    return result.strip()

def chat_infer(text):
    messages = [{"role": "user", "content": text}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.2
    )

    return processor.decode(outputs[0], skip_special_tokens=True)

# =========================
# API ROUTER
# =========================

def run_api(endpoint, image, custom_prompt, chat_text):
    try:
        if endpoint in ["bill", "invoice", "insurance"]:
            if image is None:
                return "❌ Image required"

            prompt = {
                "bill": BILL_PROMPT,
                "invoice": INVOICE_PROMPT,
                "insurance": INSURANCE_PROMPT
            }[endpoint]

            result = vision_infer(image, prompt)
            return json.dumps({"document": endpoint, "data": result}, indent=2)

        elif endpoint == "custom":
            if image is None or not custom_prompt:
                return "❌ Image + custom prompt required"

            result = vision_infer(image, custom_prompt)
            return json.dumps({"type": "custom", "data": result}, indent=2)

        elif endpoint == "chat":
            if not chat_text:
                return "❌ Chat text required"

            result = chat_infer(chat_text)
            return json.dumps({"response": result}, indent=2)

        else:
            return "❌ Invalid endpoint"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# =========================
# GRADIO UI
# =========================

with gr.Blocks(title="🧠 Multimodal Document AI (Qwen2-VL)") as demo:
    gr.Markdown("## 📄 Multimodal Document AI (Real Vision)")

    endpoint = gr.Dropdown(
        ["bill", "invoice", "insurance", "custom", "chat"],
        label="Select Mode"
    )

    image = gr.Image(type="pil", label="Upload Document Image")
    custom_prompt = gr.Textbox(label="Custom Prompt (for custom mode)")
    chat_text = gr.Textbox(label="Chat Text (for chat mode)")
    output = gr.Textbox(lines=18, label="Response (JSON)")

    run_btn = gr.Button("Run Inference 🚀")

    run_btn.click(
        run_api,
        inputs=[endpoint, image, custom_prompt, chat_text],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
