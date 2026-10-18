
import torch
from config_files.SpeechOcean762_Configs import Config
from dataloader.dataloader import data_generator
from models.model import base_Model
from models.TC import TC

configs = Config()
device = torch.device("cpu")

data_path = "./data/SpeechOcean762"

print("1. Loading datasets...")
train_dl, val_dl, test_dl = data_generator(
    data_path, configs, "supervised"
)

print("Train batches:", len(train_dl))
print("Validation batches:", len(val_dl))
print("Test batches:", len(test_dl))

print("\n2. Checking one batch...")
x, y, _, _ = next(iter(train_dl))

print("Input shape:", tuple(x.shape))
print("Label shape:", tuple(y.shape))
print("Labels in batch:", torch.unique(y).tolist())

assert x.ndim == 3, f"Expected 3D input, got {x.shape}"
assert x.shape[1] == 64, f"Expected 64 channels, got {x.shape[1]}"
assert x.shape[2] == 1536, f"Expected 1536 frames, got {x.shape[2]}"
assert y.dtype == torch.long
assert y.min().item() >= 0 and y.max().item() <= 2

print("\n3. Checking CNN...")
model = base_Model(configs).to(device)
model.eval()

with torch.no_grad():
    outputs = model(x.to(device))

print("CNN return type:", type(outputs))

if isinstance(outputs, tuple):
    print("Number of returned items:", len(outputs))
    for i, item in enumerate(outputs):
        if torch.is_tensor(item):
            print(f"Item {i} shape:", tuple(item.shape))
        else:
            print(f"Item {i} type:", type(item))
elif torch.is_tensor(outputs):
    print("CNN output shape:", tuple(outputs.shape))
else:
    raise TypeError(f"Unexpected model output: {type(outputs)}")

print("\n4. Checking temporal contrastive model...")
tc_model = TC(configs, device).to(device)
tc_model.eval()


with torch.no_grad():
    features = outputs[1]

    # TC.forward() yêu cầu hai đầu vào
    temporal_output = tc_model(features, features)

print("Temporal return type:", type(temporal_output))

if isinstance(temporal_output, tuple):
    print("Number of returned items:", len(temporal_output))

    for i, item in enumerate(temporal_output):
        if torch.is_tensor(item):
            print(f"Temporal item {i} shape:", tuple(item.shape))
        else:
            print(f"Temporal item {i} type:", type(item))

elif torch.is_tensor(temporal_output):
    print("Temporal output shape:", tuple(temporal_output.shape))

print("\nPASS: Temporal forward pass completed.")

print("Temporal return type:", type(temporal_output))

if isinstance(temporal_output, tuple):
    print("Number of returned items:", len(temporal_output))
    for i, item in enumerate(temporal_output):
        if torch.is_tensor(item):
            print(f"Temporal item {i} shape:", tuple(item.shape))
        else:
            print(f"Temporal item {i} type:", type(item))
elif torch.is_tensor(temporal_output):
    print("Temporal output shape:", tuple(temporal_output.shape))
else:
    raise TypeError(
        f"Unexpected temporal output: {type(temporal_output)}"
    )

print("\nPASS: CNN and temporal forward pass completed.")

print("\nPASS: data loader and model checks completed.")
