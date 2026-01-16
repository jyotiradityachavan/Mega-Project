import gradio as gr
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import json

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
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
Extract all key-value pairs from this BILL.
Return ONLY valid JSON.
"""

INVOICE_PROMPT = """
Extract all key-value pairs from this INVOICE.
Return ONLY valid JSON.
"""

INSURANCE_PROMPT = """
Extract all key-value pairs from this INSURANCE document.
Return ONLY valid JSON.
"""

# =========================
# INFERENCE
# =========================

def vision_infer(image, prompt):
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
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.0
    )

    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return result

# =========================
# API HANDLER
# =========================

def run(endpoint, image, custom_prompt, chat_text):
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
            return json.dumps({"type": endpoint, "data": result}, indent=2)

        elif endpoint == "custom":
            if image is None or not custom_prompt:
                return "❌ Image + Prompt required"
            result = vision_infer(image, custom_prompt)
            return json.dumps({"type": "custom", "data": result}, indent=2)

        elif endpoint == "chat":
            if not chat_text:
                return "❌ Text required"
            result = vision_infer(Image.new("RGB", (1, 1)), chat_text)
            return result

        else:
            return "❌ Invalid endpoint"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# =========================
# GRADIO UI
# =========================

with gr.Blocks(title="Multimodal Document AI (Qwen2-VL)") as demo:
    gr.Markdown("## 📄 Multimodal Document AI (Image Supported)")

    endpoint = gr.Dropdown(
        ["bill", "invoice", "insurance", "custom", "chat"],
        label="Select Mode"
    )

    image = gr.Image(type="pil", label="Upload Image")
    custom_prompt = gr.Textbox(label="Custom Prompt")
    chat_text = gr.Textbox(label="Chat Text")

    output = gr.Textbox(label="Response", lines=20)
    btn = gr.Button("Run 🚀")

    btn.click(
        run,
        inputs=[endpoint, image, custom_prompt, chat_text],
        outputs=output
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
