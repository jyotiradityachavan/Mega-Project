import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import requests

BASE_URL = "http://localhost:8000"

image_path = None

# =========================
# HELPERS
# =========================
def select_image():
    global image_path
    image_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    image_label.config(text=image_path if image_path else "No image selected")


def show_response(res):
    output.delete(1.0, tk.END)
    output.insert(tk.END, res)


# =========================
# API CALLS
# =========================
def call_root():
    r = requests.get(f"{BASE_URL}/")
    show_response(r.text)


def call_health():
    r = requests.get(f"{BASE_URL}/health")
    show_response(r.text)


def call_image_api(endpoint):
    if not image_path:
        messagebox.showerror("Error", "Image required")
        return

    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/{endpoint}",
            files={"image": f}
        )
    show_response(r.text)


def call_custom():
    if not image_path or not prompt_entry.get():
        messagebox.showerror("Error", "Image + Prompt required")
        return

    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/custom",
            files={"image": f},
            data={"prompt": prompt_entry.get()}
        )
    show_response(r.text)


def call_chat():
    if not chat_entry.get():
        messagebox.showerror("Error", "Question required")
        return

    r = requests.post(
        f"{BASE_URL}/chat",
        data={"question": chat_entry.get()}
    )
    show_response(r.text)


# =========================
# TKINTER UI
# =========================
root = tk.Tk()
root.title("🧪 Document AI – API Tester (Tkinter)")
root.geometry("850x650")

tk.Label(root, text="FastAPI Endpoint Tester", font=("Arial", 16, "bold")).pack(pady=5)

# Image picker
tk.Button(root, text="Select Image", command=select_image).pack()
image_label = tk.Label(root, text="No image selected", fg="gray")
image_label.pack(pady=3)

# Prompt
tk.Label(root, text="Custom Prompt").pack()
prompt_entry = tk.Entry(root, width=80)
prompt_entry.pack(pady=3)

# Chat
tk.Label(root, text="Chat Question").pack()
chat_entry = tk.Entry(root, width=80)
chat_entry.pack(pady=3)

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="/", width=12, command=call_root).grid(row=0, column=0, padx=4)
tk.Button(btn_frame, text="/health", width=12, command=call_health).grid(row=0, column=1, padx=4)
tk.Button(btn_frame, text="/bill", width=12, command=lambda: call_image_api("bill")).grid(row=0, column=2, padx=4)
tk.Button(btn_frame, text="/invoice", width=12, command=lambda: call_image_api("invoice")).grid(row=0, column=3, padx=4)
tk.Button(btn_frame, text="/insurance", width=12, command=lambda: call_image_api("insurance")).grid(row=0, column=4, padx=4)

tk.Button(btn_frame, text="/custom", width=12, command=call_custom).grid(row=1, column=1, pady=5)
tk.Button(btn_frame, text="/chat", width=12, command=call_chat).grid(row=1, column=3, pady=5)

# Output
tk.Label(root, text="Response").pack()
output = scrolledtext.ScrolledText(root, width=100, height=20)
output.pack(pady=5)

root.mainloop()

