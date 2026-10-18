<<<<<<< HEAD
# DINOv3 + TS-TCC for SpeechOcean762

> **Mục tiêu:** tích hợp backbone thị giác DINOv3 vào pipeline học biểu diễn chuỗi thời gian của TS-TCC, sử dụng đầu vào Log-Mel của SpeechOcean762. Tài liệu này chỉ rõ vị trí thay đổi kiến trúc, cách chạy, và sự khác nhau giữa loss gốc của TS-TCC và loss được dùng trong bản tích hợp.

## 1. Tổng quan

Trong phiên bản này, phần encoder CNN của pipeline TS-TCC được thay bằng **DINOv3 ViT-S/16 pretrained**. Module TC và pipeline huấn luyện tương phản vẫn được giữ lại về mặt ý tưởng; adapter được thêm vào để chuyển đầu vào Log-Mel thành dạng ảnh mà DINOv3 nhận được, rồi chuyển patch tokens thành đặc trưng chuỗi cho TC.

## 2. Kiến trúc sau khi tích hợp

```text
Log-Mel input (B, 64, 1536)
        │
        ▼
Resize → (224, 224), lặp 1 kênh thành 3 kênh, chuẩn hoá từng mẫu
        │
        ▼
DINOv3 ViT-S/16 backbone pretrained
        │
        ▼
Loại bỏ CLS token và 4 register tokens
        │
        ▼
196 patch tokens → lưới 14 × 14
        │
        ▼
Trung bình theo một trục lưới → chuỗi 14 bước
        │
        ▼
Feature projection: Linear(384 → 128) + LayerNorm + GELU
        │
        ├────────► features (B, 128, 14) ─────► TC module
        │
        ▼
Flatten + Dropout + Linear classifier ─────► logits (B, 3)
```

Kích thước 384 là hidden size mặc định của DINOv3 ViT-S; code lấy `hidden_size` từ cấu hình backbone khi có thể. Với ảnh 224 × 224 và patch size 16, lưới patch có kích thước 14 × 14, tương ứng 196 patch tokens.

## 3. Những file và đoạn code đã thay đổi

### 3.1 `models/dinov3_model.py` — thay encoder/backbone

Đây là file chính thể hiện việc thay encoder. Các điểm quan trọng:

**(a) Tải DINOv3 pretrained**

```python
self.backbone = AutoModel.from_pretrained(model_id)
```

`model_id` mặc định là `facebook/dinov3-vits16-pretrain-lvd1689m`. Dòng này khởi tạo backbone DINOv3 pretrained thay cho encoder CNN của bản TS-TCC gốc.

**(b) Đóng băng backbone khi cấu hình yêu cầu**

```python
if self.freeze_backbone:
    for p in self.backbone.parameters():
        p.requires_grad = False
```

Khi `dinov3_freeze_backbone=True`, trọng số DINOv3 được giữ cố định. Cách này giảm số tham số cần cập nhật và mức sử dụng VRAM trong giai đoạn huấn luyện adapter/classifier.

**(c) Chuyển Log-Mel thành đầu vào phù hợp với DINOv3**

Hàm `_to_rgb_images()` nhận tensor `(B, mel_bins, frames)`, thêm chiều kênh, resize thành `224 × 224`, lặp thành 3 kênh và chuẩn hoá theo từng mẫu. Đây là adapter tiền xử lý vì DINOv3 nhận `pixel_values` dạng ảnh.

**(d) Lấy patch tokens và tạo đặc trưng chuỗi**

```python
outputs = self.backbone(pixel_values=images)
tokens = outputs.last_hidden_state
patch_tokens = tokens[:, 1 + 4:, :]
```

Code bỏ CLS token và 4 register tokens, sau đó kiểm tra số patch tokens đúng bằng `14 × 14`. Các patch được reshape thành lưới, trung bình theo một trục và chiếu từ hidden size sang `final_out_channels` (mặc định 128):

```python
sequence = patch_grid.mean(dim=1)            # (B, 14, hidden_size)
sequence = self.feature_projection(sequence) # (B, 14, 128)
features = sequence.transpose(1, 2).contiguous()  # (B, 128, 14)
```

`features` có định dạng mà module TC hiện tại sử dụng.

### 3.2 `main_dinov3.py` — khởi tạo và chạy mô hình DINOv3 + TC

Các import thể hiện mô hình mới được nối vào pipeline:

```python
from models.dinov3_model import DINOv3_Model
from models.TC import TC
from trainer.trainer_dinov3 import Trainer_DINOv3, evaluate
```

Khởi tạo hai thành phần:

```python
model = DINOv3_Model(configs).to(device)
temporal_model = TC(configs, device).to(device)
```

Sau đó tạo optimizer cho các tham số có `requires_grad=True` của model và cho TC. Với backbone bị đóng băng, tham số của backbone không được đưa vào danh sách tham số trainable của `model_optimizer`.

Trước khi train, main thực hiện **smoke test**: lấy batch, chạy forward và ghi shape của `logits` và `features`; ở chế độ self-supervised còn kiểm tra loss có hữu hạn hay không. Sau khi train, checkpoint có validation loss tốt nhất được nạp lại và đánh giá trên test set.

### 3.3 `trainer/trainer_dinov3.py` — loss và vòng lặp huấn luyện

File này định nghĩa `nt_xent_loss`, `combined_self_supervised_loss`, `evaluate`, `save_checkpoint` và `Trainer_DINOv3`.

- `nt_xent_loss`: tính loss contrastive trên hai batch projection.
- `combined_self_supervised_loss`: cộng hai loss TC theo hai chiều cùng với NT-Xent.
- `evaluate`: dùng loss self-supervised ở chế độ `self_supervised`; dùng Cross-Entropy và accuracy cho các chế độ phân loại.
- `save_checkpoint`: lưu state dict của model, TC, optimizer và thông tin epoch/validation loss.
- `Trainer_DINOv3`: chạy epoch, kiểm tra loss/gradient, cập nhật optimizer, đánh giá validation và lưu `ckp_best.pt`, `ckp_last.pt`.

Nếu sử dụng chế độ `train_linear`, cần bảo đảm trainer có nhánh riêng để giữ `backbone` và `feature_projection` ở chế độ eval, chỉ huấn luyện `classifier`, không cập nhật TC, và lưu checkpoint trong thư mục run của `train_linear`.

## 4. Loss trước và sau khi thay DINOv3

### 4.1 Loss self-supervised của TS-TCC trong pipeline này

Về mặt mục tiêu, TS-TCC kết hợp temporal contrastive loss và contextual/instance contrastive loss. Với hai view tăng cường `aug1`, `aug2`, code gọi TC theo cả hai thứ tự:

```python
tc_loss1, projection1 = temporal_model(features1, features2)
tc_loss2, projection2 = temporal_model(features2, features1)
ntxent = nt_xent_loss(projection1, projection2, temperature=...)
loss = tc_loss1 + tc_loss2 + 0.7 * ntxent
```

Do đó loss thực tế trong `combined_self_supervised_loss()` là:

\[
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{TC}}(z_1,z_2)
+ \mathcal{L}_{\mathrm{TC}}(z_2,z_1)
+ 0.7\,\mathcal{L}_{\mathrm{NT\text{-}Xent}}(p_1,p_2)
\]

Trong đó:

- `features1`, `features2`: đặc trưng encoder từ hai view.
- `tc_loss1`, `tc_loss2`: loss temporal contrastive do module `TC` trả về.
- `projection1`, `projection2`: vector projection do TC trả về.
- `ntxent`: NT-Xent trên hai projection.
- Hệ số `0.7` là trọng số kết hợp được đặt trong code của bản tích hợp này.

Hàm NT-Xent chuẩn hoá projection theo chiều đặc trưng, ghép hai view thành `2B` vector, tính cosine similarity (tích vô hướng sau chuẩn hoá), chia cho temperature, loại đường chéo self-similarity và dùng cross-entropy để nhận diện view tương ứng của cùng mẫu.

### 4.2 Loss của bản CNN TS-TCC trước khi tích hợp

Ở bản TS-TCC CNN ban đầu, cấu trúc mục tiêu self-supervised là kết hợp các loss temporal contrastive từ module TC và loss contextual/NT-Xent từ các projection. Encoder cung cấp đặc trưng cho TC; công thức tổng hợp được dùng trong bản tích hợp này vẫn theo dạng:

\[
\mathcal{L}_{\mathrm{TS\text{-}TCC}} = \mathcal{L}_{\mathrm{TC}}^{(1)} + \mathcal{L}_{\mathrm{TC}}^{(2)} + \lambda\,\mathcal{L}_{\mathrm{NT\text{-}Xent}}
\]

Trong code tích hợp hiện tại, \(\lambda=0.7\). Không nên ghi rằng DINOv3 tự tạo ra một loss mới: **thay đổi chính là encoder/backbone và cách chuyển đổi đặc trưng**, trong khi mục tiêu contrastive được giữ theo khung TS-TCC. Nếu bản CNN gốc trong repository của bạn dùng hệ số hoặc cách tính loss khác, hãy trích dẫn trực tiếp từ `trainer/trainer.py` để mô tả chính xác khác biệt giữa hai phiên bản.

### 4.3 Loss khi supervised / linear evaluation

Ở chế độ có nhãn, code sử dụng Cross-Entropy trên logits phân loại:

\[
\mathcal{L}_{\mathrm{CE}} = -\frac{1}{B}\sum_{i=1}^{B}\sum_{c=1}^{C}
 y_{ic}\log p_{ic}
\]

Trong đó `y` là nhãn one-hot, còn `p` là xác suất dự đoán sau softmax (CrossEntropyLoss của PyTorch nhận logits và thực hiện các bước cần thiết nội bộ). Với linear evaluation, mục tiêu là đóng băng phần trích xuất đặc trưng và chỉ cập nhật lớp classifier; cần xác minh trong mã đang chạy rằng optimizer thực sự chỉ nhận tham số classifier.

## 5. Các cập nhật chính

| Thành phần | TS-TCC CNN ban đầu | Bản DINOv3 tích hợp |
|---|---|---|
| Encoder/backbone | CNN encoder của TS-TCC | DINOv3 ViT-S/16 pretrained |
| Đầu vào encoder | Chuỗi thời gian theo pipeline CNN | Log-Mel được chuyển thành ảnh 3 kênh 224 × 224 |
| Đặc trưng | Tensor đầu ra CNN | Patch tokens DINOv3, gộp thành chuỗi 14 bước |
| Chiều đặc trưng đưa vào TC | Theo cấu hình encoder CNN | `(B, 128, 14)` sau feature projection |
| TC module | Temporal contrastive module | Giữ module `TC` và điều chỉnh đầu ra encoder cho tương thích |
| Self-supervised objective | Temporal + contextual contrastive | Temporal + NT-Xent, tổng `tc_loss1 + tc_loss2 + 0.7 * ntxent` trong code tích hợp |
| Phân loại | Classification head theo encoder gốc | Dropout + Linear classifier trên đặc trưng DINOv3 |
| Pretrained weights | Không nhất thiết có backbone pretrained tương ứng | Tải từ Hugging Face; có thể đóng băng backbone |
| Kiểm tra trước train | Tuỳ bản gốc | Smoke test shape và loss trước khi train |
| Lưu mô hình | Theo cơ chế của bản gốc | `ckp_best.pt` và `ckp_last.pt` trong thư mục run |


=======

>>>>>>> fff2301b2b870f4f341d183537b0ba1b9158b4e4

