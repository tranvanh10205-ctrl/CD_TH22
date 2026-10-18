
import torch
from transformers import AutoModel

MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

print("Loading DINOv3...")
model = AutoModel.from_pretrained(MODEL_ID)
model = model.to(device).eval()

# Mô phỏng một ảnh RGB 224 x 224.
x = torch.randn(1, 3, 224, 224, device=device)

with torch.inference_mode():
    output = model(pixel_values=x)

print("Last hidden state:", output.last_hidden_state.shape)
print("Pooler output:", output.pooler_output.shape)
print("DINOv3 loaded successfully.")
