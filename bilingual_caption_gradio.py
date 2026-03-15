import torch
import torch.nn as nn
import re
import os
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from peft import get_peft_model, LoraConfig
import gradio as gr

# ======================== CONFIG ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dir = r"C:\Users\USER\Downloads\vit_embeddings-200"
checkpoint_path = r"C:\Users\USER\Documents\grad_project\final1.pt"

# ======================== CLEANING FUNCTION ========================
def clean_caption(caption):
    caption = caption.strip()
    caption = re.sub(r"\s+", " ", caption)
    caption = caption.replace("..", ".").replace("،،", "،")
    caption = re.sub(r"([?.!,،])\1+", r"\1", caption)
    caption = re.sub(r"\b(\w+)( \1\b)+", r"\1", caption)
    return caption

# ======================== MODEL ========================
class BilingualCaptioningModel(nn.Module):
    def __init__(self, mt5_model_name, mt5_hidden_size):
        super().__init__()
        self.tokenizer = MT5Tokenizer.from_pretrained(mt5_model_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<ENGLISH>", "<ARABIC>"]})
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(mt5_model_name)
        self.mt5.resize_token_embeddings(len(self.tokenizer))

        lora_config = LoraConfig(
            r=32, lora_alpha=64, target_modules=["q", "v"],
            lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
        )
        self.mt5 = get_peft_model(self.mt5, lora_config)

        self.cls_to_prompt1 = nn.Linear(768, mt5_hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.cls_to_prompt2 = nn.Linear(mt5_hidden_size, mt5_hidden_size)

    def encode_prompt(self, pixel_values):
        prompt_embed = self.cls_to_prompt1(pixel_values)
        prompt_embed = self.dropout(prompt_embed)
        prompt_embed = self.cls_to_prompt2(prompt_embed)
        return prompt_embed

    def generate_caption(self, pixel_values, lang_token):
        decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(lang_token)
        prompt_embed = self.encode_prompt(pixel_values)
        gen_ids = self.mt5.generate(
            inputs_embeds=prompt_embed,
            decoder_start_token_id=decoder_start_token_id,
            max_length=64,
            num_beams=5,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            length_penalty=1.0,
            early_stopping=True
        )
        decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [clean_caption(c) for c in decoded]

# ======================== LOAD MODEL ========================
model = BilingualCaptioningModel("google/mt5-base", 768).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)
model.eval()

# ======================== CAPTION FUNCTION ========================
def generate_caption_with_name(image, filename, lang_choice):
    if image is None or filename.strip() == "":
        return "⚠️ Please upload image and enter image file name without extension."
    
    base_name = filename.strip()
    pt_path = os.path.join(embedding_dir, base_name + ".pt")

    if not os.path.exists(pt_path):
        return f"⚠️ No embedding file found for '{base_name}.pt' in embeddings folder."

    embedding = torch.load(pt_path).unsqueeze(0).to(device).float()

    with torch.no_grad():
        if lang_choice == "Arabic":
            caption = model.generate_caption(embedding, "<ARABIC>")[0]
            return f"🇸🇦 Arabic: {caption}"
        else:
            caption = model.generate_caption(embedding, "<ENGLISH>")[0]
            return f"🇬🇧 English: {caption}"

# ======================== CUSTOM GREEN THEME ========================
custom_css = """
body {
    background-color: #e6f2e6 !important;
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3, label, p {
    color: #1b5e20 !important;
    font-weight: bold;
}

.gr-button {
    background: linear-gradient(135deg, #43a047, #2e7d32) !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    box-shadow: 0px 3px 6px rgba(0,0,0,0.2) !important;
}

input, textarea, .gr-textbox, .gr-radio, .gr-image {
    border: 2px solid #81c784 !important;
    border-radius: 8px !important;
    background-color: #f1f8e9 !important;
    color: #1b5e20 !important;
}

.gr-textbox textarea {
    font-weight: bold !important;
    font-size: 16px !important;
}
"""

# ======================== GRADIO INTERFACE ========================
with gr.Blocks(css=custom_css) as iface:
    gr.Markdown("## 🌿 Bilingual Image Captioning Tool")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="📤 Upload Image")
            filename_input = gr.Textbox(label="📝 Enter Image File Name (without extension)", placeholder="e.g. img_1")
            lang_choice = gr.Radio(choices=["Arabic", "English"], value="Arabic", label="🌐 Select Language")
            submit_btn = gr.Button("🎯 Generate Caption")

        with gr.Column():
            caption_output = gr.Textbox(label="🖼️ Generated Caption", lines=4)

    submit_btn.click(fn=generate_caption_with_name,
                     inputs=[image_input, filename_input, lang_choice],
                     outputs=caption_output)

iface.launch()