
import os
import argparse
import random
import logging
import numpy as np
import torch

from config_files.SpeechOcean762_DINOv3_Configs import Config
from dataloader.dataloader import data_generator
from models.dinov3_model import DINOv3_Model
from models.TC import TC
from trainer.trainer_dinov3 import Trainer_DINOv3, evaluate


def main():
    parser = argparse.ArgumentParser(
        description="DINOv3 + TS-TCC for SpeechOcean762"
    )
    parser.add_argument(
        "--training_mode",
        choices=["self_supervised", "supervised", "train_linear"],
        default="self_supervised"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_path", default="./data/SpeechOcean762")
    parser.add_argument(
        "--logs_save_dir",
        default="experiments_logs/SpeechOcean762_DINOv3_TS-TCC"
    )
    parser.add_argument("--experiment_description", default="Exp1")
    parser.add_argument("--run_description", default="DINOv3")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Bạn chọn CUDA nhưng PyTorch không phát hiện GPU CUDA."
        )

    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    configs = Config()

    run_dir = os.path.join(
        args.logs_save_dir,
        args.experiment_description,
        args.run_description,
        f"{args.training_mode}_seed_{args.seed}"
    )
    os.makedirs(run_dir, exist_ok=True)

    logger = logging.getLogger("DINOv3_TS-TCC")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        os.path.join(run_dir, "training.log"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Device: {device}")
    logger.info(f"Mode: {args.training_mode}")
    logger.info(f"Data path: {args.data_path}")

    train_dl, valid_dl, test_dl = data_generator(
        args.data_path, configs, args.training_mode
    )

    logger.info(
        f"Train batches={len(train_dl)}, "
        f"validation batches={len(valid_dl)}, "
        f"test batches={len(test_dl)}"
    )

    model = DINOv3_Model(configs).to(device)
    temporal_model = TC(configs, device).to(device)

    # Linear evaluation loads the self-supervised checkpoint first.
    if args.training_mode == "train_linear":
        pretrained_path = os.path.join(
            args.logs_save_dir,
            args.experiment_description,
            args.run_description,
            f"self_supervised_seed_{args.seed}",
            "saved_models",
            "ckp_last.pt"
        )

        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                "Không tìm thấy checkpoint self-supervised:\n"
                f"{pretrained_path}\n"
                "Hãy chạy self_supervised trước với cùng seed, "
                "experiment_description và run_description."
            )

        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if "temporal_contr_model_state_dict" in checkpoint:
            temporal_model.load_state_dict(
                checkpoint["temporal_contr_model_state_dict"]
            )

        # Freeze backbone and feature projection.
        for parameter in model.parameters():
            parameter.requires_grad = False

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

        model.freeze_backbone = True
        model.backbone.eval()

        logger.info(f"Loaded pretrained checkpoint: {pretrained_path}")
        logger.info("Linear evaluation: only classifier parameters train.")

    trainable_model_params = [
        p for p in model.parameters() if p.requires_grad
    ]

    if not trainable_model_params:
        raise RuntimeError("Model không có tham số nào được phép huấn luyện.")

    model_optimizer = torch.optim.AdamW(
        trainable_model_params,
        lr=configs.lr,
        betas=(configs.beta1, configs.beta2),
        weight_decay=3e-4
    )

    temporal_optimizer = torch.optim.AdamW(
        temporal_model.parameters(),
        lr=configs.lr,
        betas=(configs.beta1, configs.beta2),
        weight_decay=3e-4
    )

    # Smoke test before training.
    batch = next(iter(train_dl))
    _, labels, aug1, aug2 = batch

    logger.info(
        f"Smoke batch: labels={tuple(labels.shape)}, "
        f"aug1={tuple(aug1.shape)}, aug2={tuple(aug2.shape)}"
    )

    model.eval()
    temporal_model.eval()

    with torch.no_grad():
        test_x = aug1[:min(2, len(aug1))].float().to(device)
        logits, features = model(test_x)

        logger.info(
            f"Smoke forward: logits={tuple(logits.shape)}, "
            f"features={tuple(features.shape)}"
        )

    if args.training_mode == "self_supervised":
        from trainer.trainer_dinov3 import combined_self_supervised_loss

        with torch.no_grad():
            loss, _ = combined_self_supervised_loss(
                model,
                temporal_model,
                aug1[:min(2, len(aug1))].float().to(device),
                aug2[:min(2, len(aug2))].float().to(device),
                configs
            )

        if not torch.isfinite(loss):
            raise RuntimeError("Smoke test phát hiện loss không hữu hạn.")

        logger.info(f"Smoke self-supervised loss={loss.item():.4f}")

    logger.info("Starting training...")

    Trainer_DINOv3(
        model, temporal_model,
        model_optimizer, temporal_optimizer,
        train_dl, valid_dl, device, logger, configs,
        run_dir, args.training_mode
    )

    # Evaluate the best validation checkpoint on test data.
        # Evaluate the best validation checkpoint on test data.
    best_path = os.path.join(run_dir, "saved_models", "ckp_best.pt")

    if not os.path.isfile(best_path):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint sau huấn luyện: {best_path}. "
            "Hãy kiểm tra Trainer_DINOv3 có lưu ckp_best.pt hay không."
        )

    checkpoint = torch.load(best_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    # Temporal model may not be trained in train_linear mode.
    if "temporal_contr_model_state_dict" in checkpoint:
        temporal_model.load_state_dict(
            checkpoint["temporal_contr_model_state_dict"]
        )

    test_loss, test_acc = evaluate(
        model, temporal_model, test_dl, device,
        args.training_mode, configs
    )

    logger.info(
        f"Best epoch={checkpoint.get('epoch', 'unknown')} | "
        f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f}"
    )


if __name__ == "__main__":
    main()
