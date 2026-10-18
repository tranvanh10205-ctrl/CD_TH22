import torch
from config_files.SpeechOcean762_DINOv3_Configs import Config
from dataloader.dataloader import data_generator
from models.dinov3_model import DINOv3_Model
from models.TC import TC

configs = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

data_path = "./data/SpeechOcean762"
train_dl, val_dl, test_dl = data_generator(
    data_path, configs, "self_supervised"
)

x, y, aug1, aug2 = next(iter(train_dl))
print("Input:", tuple(x.shape))
print("Augmentation 1:", tuple(aug1.shape))
print("Augmentation 2:", tuple(aug2.shape))

model = DINOv3_Model(configs).to(device)
tc_model = TC(configs, device).to(device)

model.eval()
tc_model.eval()

with torch.no_grad():
    logits1, features1 = model(aug1.to(device))
    logits2, features2 = model(aug2.to(device))

    print("DINOv3 logits:", tuple(logits1.shape))
    print("DINOv3 features:", tuple(features1.shape))

    tc_loss_1, projection_1 = tc_model(features1, features2)
    tc_loss_2, projection_2 = tc_model(features2, features1)

    print("TC loss 1:", tuple(tc_loss_1.shape))
    print("TC projection 1:", tuple(projection_1.shape))
    print("TC loss 2:", tuple(tc_loss_2.shape))
    print("TC projection 2:", tuple(projection_2.shape))

assert logits1.shape == (x.shape[0], configs.num_classes)
assert features1.shape == (x.shape[0], 128, 14)
assert projection_1.shape == (x.shape[0], 32)
assert projection_2.shape == (x.shape[0], 32)

print("\nPASS: DINOv3 + TS-TCC forward test completed.")
