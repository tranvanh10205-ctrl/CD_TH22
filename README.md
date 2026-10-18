## 4. Hàm mất mát (Loss Functions)

### 4.1. Loss tự giám sát trong phiên bản DINOv3 + TS-TCC

Trong quá trình học tự giám sát, mô hình nhận hai view tăng cường của cùng một mẫu đầu vào. Backbone DINOv3 trích xuất đặc trưng, sau đó các đặc trưng được đưa vào module Temporal Contrasting (TC). Các vector projection được sử dụng để tính loss NT-Xent.

Theo cấu trúc được mô tả trong trainer, loss tổng có dạng:

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{TC}}^{(1)}
+
\mathcal{L}_{\text{TC}}^{(2)}
+
\lambda \mathcal{L}_{\text{NT-Xent}}
$$

Trong đó:

$$
\mathcal{L}_{\text{TC}}^{(1)}
=
\operatorname{TC}(f_1,f_2)
$$

$$
\mathcal{L}_{\text{TC}}^{(2)}
=
\operatorname{TC}(f_2,f_1)
$$

Với hệ số được mô tả trong phiên bản hiện tại:

$$
\lambda = 0.7
$$

Do đó, loss tổng được viết thành:

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{TC}}^{(1)}
+
\mathcal{L}_{\text{TC}}^{(2)}
+
0.7\mathcal{L}_{\text{NT-Xent}}
$$

Ý nghĩa các ký hiệu:

- \(f_1\), \(f_2\): đặc trưng thu được từ hai view tăng cường.
- \(\mathcal{L}_{\text{TC}}^{(1)}\): loss của TC khi nhận hai view theo thứ tự thứ nhất.
- \(\mathcal{L}_{\text{TC}}^{(2)}\): loss của TC khi đảo thứ tự hai view.
- \(\mathcal{L}_{\text{NT-Xent}}\): loss tương phản tính trên các vector projection.
- \(\lambda\): hệ số điều chỉnh mức đóng góp của NT-Xent vào loss tổng.

**Lưu ý về tính chính xác:** Công thức trên mô tả cấu trúc kết hợp các thành phần loss theo thiết kế được trình bày. Để khẳng định chính xác phép tính trong chương trình, cần đối chiếu hàm `combined_self_supervised_loss()` trong `trainer/trainer_dinov3.py` và phần triển khai của `TC` trong `models/TC.py`.

### 4.2. NT-Xent (Normalized Temperature-scaled Cross-Entropy)

NT-Xent là hàm mất mát tương phản. Mục tiêu của nó là đưa các biểu diễn của cùng một mẫu lại gần nhau, đồng thời phân biệt chúng với biểu diễn của các mẫu khác.

Giả sử batch ban đầu có \(N\) mẫu. Sau khi tạo hai view cho mỗi mẫu, ta có \(2N\) vector projection.

Gọi \(u_i\) là vector projection trước khi chuẩn hóa. Vector sau chuẩn hóa L2 được tính như sau:

$$
z_i = \frac{u_i}{\|u_i\|_2}
$$

Độ tương đồng giữa hai vector chuẩn hóa được tính bằng tích vô hướng:

$$
s(z_i,z_j) = z_i^\top z_j
$$

Vì các vector đã được chuẩn hóa L2, giá trị này tương đương độ tương đồng cosine.

Gọi \(p(i)\) là chỉ số của vector thuộc view còn lại nhưng được tạo từ cùng một mẫu với vector \(z_i\). Loss của vector neo \(z_i\) được xác định bởi:

$$
\ell_i =
-\log
\left(
\frac{
\exp\left(s(z_i,z_{p(i)})/\tau\right)
}{
\sum_{\substack{k=1\\k\ne i}}^{2N}
\exp\left(s(z_i,z_k)/\tau\right)
}
\right)
$$

Loss NT-Xent trung bình trên \(2N\) vector là:

$$
\mathcal{L}_{\text{NT-Xent}}
=
\frac{1}{2N}
\sum_{i=1}^{2N}\ell_i
$$

Trong đó:

- \(z_i\): vector projection đã chuẩn hóa.
- \(z_{p(i)}\): vector dương, thuộc view còn lại của cùng mẫu.
- \(s(z_i,z_j)\): độ tương đồng giữa hai vector.
- \(\tau\): hệ số nhiệt độ (temperature), với \(\tau > 0\).
- \(k \ne i\): loại vector neo khỏi mẫu số.

Công thức trên giả định cách triển khai NT-Xent tiêu chuẩn: mỗi vector có một vector dương tương ứng, còn các vector khác trong batch đóng vai trò đối tượng so sánh. Khi đối chiếu với code, cần kiểm tra cách ghép hai view, cách xác định cặp dương và cách loại bỏ đường chéo của ma trận tương đồng.

### 4.3. Loss trong TS-TCC gốc

TS-TCC (Time-Series Representation Learning via Temporal and Contextual Contrasting) sử dụng hai hướng học biểu diễn bổ trợ:

1. **Temporal contrasting:** học quan hệ thời gian bằng cách dự đoán biểu diễn ở các bước tương lai từ thông tin ngữ cảnh và phân biệt biểu diễn tương lai đúng với các biểu diễn âm.
2. **Contextual contrasting:** học biểu diễn có tính phân biệt thông qua mục tiêu tương phản ở mức ngữ cảnh giữa các view.

Ở mức khái niệm, hai thành phần có thể được ký hiệu là:

$$
\mathcal{L}_{\text{TS-TCC}}
=
\mathcal{L}_{\text{temporal}}
+
\mathcal{L}_{\text{contextual}}
$$

Đây là biểu diễn khái quát cho hai mục tiêu học, không phải công thức triển khai chính xác được khẳng định cho mọi phiên bản TS-TCC. Cách kết hợp, hệ số và các phép tính cụ thể cần được xác nhận từ mã nguồn phiên bản gốc đang sử dụng.

Trong phiên bản tích hợp DINOv3, cần phân biệt ba khía cạnh:

- **Thay đổi kiến trúc:** thay encoder CNN bằng DINOv3 ViT-S/16 pretrained, thêm bước tiền xử lý đầu vào và chuyển patch tokens thành đặc trưng chuỗi.
- **Thay đổi mục tiêu tối ưu:** chỉ kết luận có thay đổi khi đối chiếu công thức loss và mã nguồn của hai phiên bản.
- **Loss thực tế:** được xác định bởi các phép tính trong `combined_self_supervised_loss()` và module `TC`.

Không nên kết luận rằng loss của bản DINOv3 hoàn toàn giống loss của TS-TCC gốc nếu chưa kiểm tra cả hai triển khai.

### 4.4. Cross-Entropy trong phân loại có giám sát

Trong các chế độ phân loại có nhãn, mô hình tạo ra một vector logits cho mỗi mẫu. Gọi \(a_{ic}\) là logit của mẫu \(i\) tại lớp \(c\), \(C\) là số lớp và \(y_i\) là nhãn đúng của mẫu \(i\).

Xác suất dự đoán cho lớp \(c\) được tính bằng softmax:

$$
p_{ic}
=
\frac{\exp(a_{ic})}
{\sum_{j=1}^{C}\exp(a_{ij})}
$$

Với batch có \(B\) mẫu, Cross-Entropy trung bình là:

$$
\mathcal{L}_{\text{CE}}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log p_{i,y_i}
$$

Trong đó \(p_{i,y_i}\) là xác suất dự đoán cho lớp đúng của mẫu \(i\).

Trong PyTorch, `torch.nn.CrossEntropyLoss` nhận trực tiếp logits và nhãn lớp dưới dạng chỉ số nguyên. Không cần áp dụng softmax trước khi truyền logits vào hàm loss.

Đối với chế độ linear evaluation, mục tiêu thông thường là giữ cố định backbone và chỉ huấn luyện classifier trên các đặc trưng đã trích xuất. Tuy nhiên, cần kiểm tra `requires_grad`, trạng thái `train()`/`eval()` và danh sách tham số của optimizer để xác nhận chương trình thực hiện đúng chế độ này.
