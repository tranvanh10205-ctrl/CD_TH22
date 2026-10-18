import numpy as np
import torch


def DataTransform(sample, config):

    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)



def permutation(x, max_segments=5, seg_mode="random"):
    """
    x: Tensor có shape (batch, channels, time)
    Hoán vị các đoạn theo trục thời gian, giữ nguyên batch và channels.
    """
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
        is_tensor = True
        original_device = x.device
        original_dtype = x.dtype
    else:
        x_np = np.asarray(x)
        is_tensor = False

    batch_size, channels, seq_len = x_np.shape

    if max_segments <= 1 or seq_len <= 1:
        result = x_np.copy()
    else:
        result = np.empty_like(x_np)

        for i in range(batch_size):
            n_segments = np.random.randint(2, min(max_segments, seq_len) + 1)

            # Chia trục thời gian thành các đoạn liên tiếp,
            # độ dài có thể khác nhau.
            split_points = np.sort(
                np.random.choice(
                    np.arange(1, seq_len),
                    size=n_segments - 1,
                    replace=False
                )
            )
            boundaries = np.concatenate(([0], split_points, [seq_len]))
            segments = [
                x_np[i, :, boundaries[j]:boundaries[j + 1]]
                for j in range(n_segments)
            ]

            # Hoán vị thứ tự các đoạn, không hoán vị batch/channel.
            order = np.random.permutation(n_segments)
            result[i] = np.concatenate(
                [segments[j] for j in order],
                axis=-1
            )

    if is_tensor:
        return torch.as_tensor(
            result, device=original_device, dtype=original_dtype
        )
    return result


