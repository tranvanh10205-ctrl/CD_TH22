<<<<<<< HEAD
# DINOv3 + TS-TCC 

> **Mục tiêu:** tích hợp backbone thị giác DINOv3 vào pipeline học biểu diễn chuỗi thời gian của TS-TCC, sử dụng đầu vào Log-Mel của SpeechOcean762. Tài liệu này chỉ rõ vị trí thay đổi kiến trúc, cách chạy, và sự khác nhau giữa loss gốc của TS-TCC và loss được dùng trong bản tích hợp.

## 1. Tổng quan

Trong phiên bản này, phần encoder CNN của pipeline TS-TCC được thay bằng **DINOv3 ViT-S/16 pretrained**. Module TC và pipeline huấn luyện tương phản vẫn được giữ lại về mặt ý tưởng; adapter được thêm vào để chuyển đầu vào Log-Mel thành dạng ảnh mà DINOv3 nhận được, rồi chuyển patch tokens thành đặc trưng chuỗi cho TC.

## 2. Kiến trúc sau khi tích hợp

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


Kích thước 384 là hidden size mặc định của DINOv3 ViT-S; code lấy hidden_size từ cấu hình backbone khi có thể. Với ảnh 224 × 224 và patch size 16, lưới patch có kích thước 14 × 14, tương ứng 196 patch tokens.

## 3. Những file và đoạn code đã thay đổi

### 3.1 models/dinov3_model.py — thay encoder/backbone

Đây là file chính thể hiện việc thay encoder. Các điểm quan trọng:

**(a) Tải DINOv3 pretrained**

self.backbone = AutoModel.from_pretrained(model_id)


model_id mặc định là facebook/dinov3-vits16-pretrain-lvd1689m. Dòng này khởi tạo backbone DINOv3 pretrained thay cho encoder CNN của bản TS-TCC gốc.

**(b) Đóng băng backbone khi cấu hình yêu cầu**

if self.freeze_backbone:
    for p in self.backbone.parameters():
        p.requires_grad = False


Khi dinov3_freeze_backbone=True, trọng số DINOv3 được giữ cố định. Cách này giảm số tham số cần cập nhật và mức sử dụng VRAM trong giai đoạn huấn luyện adapter/classifier.

**(c) Chuyển Log-Mel thành đầu vào phù hợp với DINOv3**

Hàm _to_rgb_images() nhận tensor (B, mel_bins, frames), thêm chiều kênh, resize thành 224 × 224, lặp thành 3 kênh và chuẩn hoá theo từng mẫu. Đây là adapter tiền xử lý vì DINOv3 nhận pixel_values dạng ảnh.

**(d) Lấy patch tokens và tạo đặc trưng chuỗi**

outputs = self.backbone(pixel_values=images)
tokens = outputs.last_hidden_state
patch_tokens = tokens[:, 1 + 4:, :]


Code bỏ CLS token và 4 register tokens, sau đó kiểm tra số patch tokens đúng bằng 14 × 14. Các patch được reshape thành lưới, trung bình theo một trục và chiếu từ hidden size sang final_out_channels (mặc định 128):

sequence = patch_grid.mean(dim=1)            # (B, 14, hidden_size)
sequence = self.feature_projection(sequence) # (B, 14, 128)
features = sequence.transpose(1, 2).contiguous()  # (B, 128, 14)


features có định dạng mà module TC hiện tại sử dụng.

### 3.2 main_dinov3.py — khởi tạo và chạy mô hình DINOv3 + TC

Các import thể hiện mô hình mới được nối vào pipeline:

from models.dinov3_model import DINOv3_Model
from models.TC import TC
from trainer.trainer_dinov3 import Trainer_DINOv3, evaluate


Khởi tạo hai thành phần:

model = DINOv3_Model(configs).to(device)
temporal_model = TC(configs, device).to(device)


Sau đó tạo optimizer cho các tham số có requires_grad=True của model và cho TC. Với backbone bị đóng băng, tham số của backbone không được đưa vào danh sách tham số trainable của model_optimizer.

Trước khi train, main thực hiện **smoke test**: lấy batch, chạy forward và ghi shape của logits và features; ở chế độ self-supervised còn kiểm tra loss có hữu hạn hay không. Sau khi train, checkpoint có validation loss tốt nhất được nạp lại và đánh giá trên test set.

### 3.3 trainer/trainer_dinov3.py — loss và vòng lặp huấn luyện

File này định nghĩa nt_xent_loss, combined_self_supervised_loss, evaluate, save_checkpoint và Trainer_DINOv3.

- nt_xent_loss: tính loss contrastive trên hai batch projection.
- combined_self_supervised_loss: cộng hai loss TC theo hai chiều cùng với NT-Xent.
- evaluate: dùng loss self-supervised ở chế độ self_supervised; dùng Cross-Entropy và accuracy cho các chế độ phân loại.
- save_checkpoint: lưu state dict của model, TC, optimizer và thông tin epoch/validation loss.
- Trainer_DINOv3: chạy epoch, kiểm tra loss/gradient, cập nhật optimizer, đánh giá validation và lưu ckp_best.pt, ckp_last.pt.

Nếu sử dụng chế độ train_linear, cần bảo đảm trainer có nhánh riêng để giữ backbone và feature_projection ở chế độ eval, chỉ huấn luyện classifier, không cập nhật TC, và lưu checkpoint trong thư mục run của train_linear.


## 4. Hàm mất mát (Loss Functions)

### 4.1. Loss tự giám sát trong phiên bản DINOv3 + TS-TCC

Trong phiên bản tích hợp này, backbone DINOv3 trích xuất đặc trưng từ hai view tăng cường của cùng một mẫu đầu vào. Đặc trưng được đưa vào module Temporal Contrasting (TC), đồng thời các vector projection được sử dụng để tính NT-Xent.

Theo cách triển khai được mô tả trong trainer/trainer_dinov3.py, hàm mất mát tổng có dạng:

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{TC}}^{(1)}
+
\mathcal{L}_{\mathrm{TC}}^{(2)}
+
\lambda \mathcal{L}_{\mathrm{NT\text{-}Xent}}
$$

Trong đó:

$$
\mathcal{L}_{\mathrm{TC}}^{(1)}
=
\mathrm{TC}(f_1,f_2)
$$

$$
\mathcal{L}_{\mathrm{TC}}^{(2)}
=
\mathrm{TC}(f_2,f_1)
$$

Với hệ số được sử dụng trong code hiện tại:

$$
\lambda = 0.7
$$

Do đó:

$$
\boxed{
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{TC}}^{(1)}
+
\mathcal{L}_{\mathrm{TC}}^{(2)}
+
0.7\mathcal{L}_{\mathrm{NT\text{-}Xent}}
}
$$

Các ký hiệu được hiểu như sau:

- $f_1, f_2$: đặc trưng chuỗi thời gian thu được từ hai view tăng cường.
- $\mathcal{L}_{\mathrm{TC}}^{(1)}$: temporal contrastive loss khi TC nhận hai view theo thứ tự thứ nhất.
- $\mathcal{L}_{\mathrm{TC}}^{(2)}$: temporal contrastive loss khi đảo thứ tự hai view.
- $\mathcal{L}_{\mathrm{NT\text{-}Xent}}$: loss tương phản giữa các vector projection của hai view.
- $\lambda$: hệ số điều chỉnh mức đóng góp của NT-Xent trong loss tổng.

**Lưu ý:** Công thức trên mô tả cách kết hợp các thành phần loss. Công thức chi tiết của từng temporal contrastive loss phải được xác định từ phần triển khai trong models/TC.py; không nên đồng nhất nó với NT-Xent hoặc tự giả định cách tính bên trong module TC.

### 4.2. Công thức NT-Xent

NT-Xent (Normalized Temperature-scaled Cross-Entropy) là hàm mất mát tương phản, khuyến khích hai biểu diễn của cùng một mẫu gần nhau trong không gian đặc trưng, đồng thời phân biệt chúng với biểu diễn của các mẫu khác.

Giả sử một batch có $N$ mẫu ban đầu. Sau khi tạo hai view, ta có $2N$ vector biểu diễn đã được chuẩn hóa:

$$
z_i = \frac{u_i}{\|u_i\|_2}
$$

Trong đó $u_i$ là vector projection trước chuẩn hóa và $z_i$ là vector sau chuẩn hóa L2.

Độ tương đồng cosine giữa hai vector được tính bằng:

$$
s(z_i,z_j)=z_i^\top z_j
$$

Gọi $p(i)$ là chỉ số của vector thuộc view còn lại nhưng xuất phát từ cùng một mẫu với $z_i$. Loss cho một vector neo (anchor) được xác định bởi:

$$
\ell_i =
-\log
\frac{
\exp\left(s(z_i,z_{p(i)})/\tau\right)
}{
\displaystyle
\sum_{\substack{k=1\\k\ne i}}^{2N}
\exp\left(s(z_i,z_k)/\tau\right)
}
$$

Loss NT-Xent trung bình trên toàn bộ $2N$ vector là:

$$
\boxed{
\mathcal{L}_{\mathrm{NT\text{-}Xent}}
=
\frac{1}{2N}
\sum_{i=1}^{2N}\ell_i
}
$$

Trong đó:

- $z_i$: vector projection đã chuẩn hóa.
- $z_{p(i)}$: vector dương (positive pair), thuộc view còn lại của cùng mẫu.
- $s(z_i,z_j)$: độ tương đồng cosine giữa hai vector.
- $\tau > 0$: hệ số nhiệt độ (temperature), điều chỉnh độ sắc nét của phân phối tương đồng.
- Các vector còn lại trong mẫu số đóng vai trò đối tượng so sánh âm đối với anchor đang xét.

Công thức này giả định cách tính NT-Xent chuẩn với một positive pair cho mỗi anchor và tất cả vector còn lại, ngoại trừ chính anchor, nằm trong mẫu số. Khi mô tả chính xác code, cần bảo đảm cách ghép batch, xác định positive pair và loại bỏ đường chéo của ma trận tương đồng phù hợp với công thức này.

### 4.3. Loss của TS-TCC CNN ban đầu

TS-TCC (Time-Series Representation Learning via Temporal and Contextual Contrasting) sử dụng hai mục tiêu tương phản bổ trợ:

1. **Temporal contrasting:** học biểu diễn có khả năng nắm bắt quan hệ thời gian bằng cách dự đoán biểu diễn ở các bước tương lai từ thông tin ngữ cảnh, đồng thời phân biệt biểu diễn tương lai đúng với các mẫu âm.
2. **Contextual contrasting:** học biểu diễn có tính phân biệt giữa các mẫu và các view tăng cường thông qua mục tiêu tương phản ở mức ngữ cảnh.

Vì temporal contrasting và contextual contrasting có cơ chế tính loss riêng, không nên coi toàn bộ loss của TS-TCC gốc là một công thức NT-Xent duy nhất.

Ở mức khái quát, mục tiêu có thể được biểu diễn như sau:

$$
\mathcal{L}_{\mathrm{TS\text{-}TCC}}
=
\mathcal{L}_{\mathrm{temporal}}
+
\mathcal{L}_{\mathrm{contextual}}
$$

Đây là cách biểu diễn hai thành phần mục tiêu ở mức khái niệm, không phải khẳng định rằng mọi bản triển khai TS-TCC đều dùng đúng phép cộng với hệ số bằng 1. Công thức tổng hợp, hệ số và cách triển khai cụ thể cần được đối chiếu với mã nguồn của phiên bản CNN gốc đang sử dụng.

Trong bản DINOv3 tích hợp, encoder CNN được thay bằng backbone DINOv3 ViT-S/16 pretrained và đặc trưng patch được chuyển thành chuỗi trước khi đưa vào TC. Vì vậy, cần phân biệt:

- **Thay đổi kiến trúc:** backbone, tiền xử lý đầu vào và phép chiếu đặc trưng.
- **Thay đổi mục tiêu tối ưu:** chỉ được kết luận khi đối chiếu cách tính loss trong trainer và module TC của cả hai phiên bản.
- **Loss thực tế của bản tích hợp:** được xác định bởi các phép tính trong combined_self_supervised_loss() và phần triển khai của TC.

Không nên khẳng định loss của bản tích hợp hoàn toàn giống loss gốc nếu chưa kiểm tra hai triển khai.

### 4.4. Cross-Entropy cho phân loại có giám sát

Trong các chế độ phân loại có nhãn, mô hình dự đoán logits cho $C$ lớp. Gọi $a_{ic}$ là logit của mẫu $i$ tại lớp $c$, còn $y_i$ là nhãn đúng của mẫu đó.

Xác suất dự đoán được tính bằng softmax:

$$
p_{ic}
=
\frac{\exp(a_{ic})}
{\displaystyle\sum_{j=1}^{C}\exp(a_{ij})}
$$

Cross-Entropy trung bình trên batch kích thước $B$ là:

$$
\boxed{
\mathcal{L}_{\mathrm{CE}}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log p_{i,y_i}
}
$$

Trong đó $p_{i,y_i}$ là xác suất mô hình gán cho lớp đúng của mẫu $i$.

Trong PyTorch, torch.nn.CrossEntropyLoss nhận trực tiếp logits và nhãn lớp dạng chỉ số nguyên; không cần áp dụng softmax trước khi gọi hàm loss.

Đối với linear evaluation, mục tiêu thường là đóng băng backbone và chỉ huấn luyện classifier trên các đặc trưng đã trích xuất. Tuy nhiên, cần kiểm tra cấu hình requires_grad, trạng thái train/eval và danh sách tham số của optimizer để xác nhận chế độ này được triển khai đúng trong code.
