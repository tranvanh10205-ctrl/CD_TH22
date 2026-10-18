
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature=0.2):
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError("NT-Xent requires tensors shaped (B, D).")
    if z1.shape != z2.shape:
        raise ValueError(f"Projection shape mismatch: {z1.shape}, {z2.shape}")

    batch_size = z1.shape[0]
    if batch_size < 2:
        raise ValueError("NT-Xent requires at least 2 samples per batch.")

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)

    similarity = representations @ representations.T
    similarity = similarity / temperature

    total = 2 * batch_size
    diagonal = torch.eye(total, device=z1.device, dtype=torch.bool)
    similarity = similarity.masked_fill(diagonal, float("-inf"))

    targets = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z1.device),
        torch.arange(0, batch_size, device=z1.device)
    ])

    return F.cross_entropy(similarity, targets)


def combined_self_supervised_loss(
    model, temporal_model, aug1, aug2, configs
):
    _, features1 = model(aug1)
    _, features2 = model(aug2)

    features1 = F.normalize(features1, dim=1)
    features2 = F.normalize(features2, dim=1)

    tc_loss1, projection1 = temporal_model(features1, features2)
    tc_loss2, projection2 = temporal_model(features2, features1)

    ntxent = nt_xent_loss(
        projection1,
        projection2,
        temperature=configs.Context_Cont.temperature
    )

    loss = tc_loss1 + tc_loss2 + 0.7 * ntxent

    return loss, {
        "tc_loss1": tc_loss1.detach(),
        "tc_loss2": tc_loss2.detach(),
        "ntxent": ntxent.detach(),
    }


def evaluate(
    model, temporal_model, loader, device, training_mode, configs
):
    model.eval()
    temporal_model.eval()
    criterion = nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels, aug1, aug2 in loader:
            data = data.float().to(device)
            labels = labels.long().to(device)
            aug1 = aug1.float().to(device)
            aug2 = aug2.float().to(device)

            if training_mode == "self_supervised":
                loss, _ = combined_self_supervised_loss(
                    model, temporal_model, aug1, aug2, configs
                )
                batch_loss = loss.item()
            else:
                logits, _ = model(data)
                loss = criterion(logits, labels)
                batch_loss = loss.item()
                correct += (
                    logits.argmax(dim=1) == labels
                ).sum().item()

            batch_size = data.shape[0]
            loss_sum += batch_loss * batch_size
            total += batch_size

    mean_loss = loss_sum / max(total, 1)
    accuracy = (
        correct / max(total, 1)
        if training_mode != "self_supervised" else 0.0
    )

    return mean_loss, accuracy


def save_checkpoint(
    path, epoch, model, temporal_model,
    model_optimizer, temporal_optimizer, best_valid_loss
):
    torch.save({
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
        "model_state_dict": model.state_dict(),
        "temporal_contr_model_state_dict": temporal_model.state_dict(),
        "model_optimizer_state_dict": model_optimizer.state_dict(),
        "temporal_optimizer_state_dict": temporal_optimizer.state_dict(),
    }, path)


def Trainer_DINOv3(
    model, temporal_model, model_optimizer, temporal_optimizer,
    train_dl, valid_dl, device, logger, configs,
    experiment_log_dir, training_mode
):
    allowed_modes = ("self_supervised", "supervised", "train_linear")
    if training_mode not in allowed_modes:
        raise ValueError(f"Unsupported training mode: {training_mode}")

    saved_dir = os.path.join(experiment_log_dir, "saved_models")
    os.makedirs(saved_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        model_optimizer, mode="min", patience=3, factor=0.5
    )

    best_valid_loss = float("inf")
    best_path = os.path.join(saved_dir, "ckp_best.pt")
    last_path = os.path.join(saved_dir, "ckp_last.pt")

    for epoch in range(1, configs.num_epoch + 1):
        model.train()
        temporal_model.train()

        # train_linear must not update the frozen feature extractor.
        if training_mode == "train_linear":
            model.backbone.eval()
            model.feature_projection.eval()
            model.classifier.train()
            temporal_model.eval()

        loss_sum = 0.0
        correct = 0
        total = 0

        for data, labels, aug1, aug2 in train_dl:
            data = data.float().to(device)
            labels = labels.long().to(device)
            aug1 = aug1.float().to(device)
            aug2 = aug2.float().to(device)

            model_optimizer.zero_grad(set_to_none=True)
            temporal_optimizer.zero_grad(set_to_none=True)

            if training_mode == "self_supervised":
                loss, _ = combined_self_supervised_loss(
                    model, temporal_model, aug1, aug2, configs
                )
            else:
                logits, _ = model(data)
                loss = criterion(logits, labels)
                correct += (
                    logits.argmax(dim=1) == labels
                ).sum().item()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}: {loss.item()}"
                )

            loss.backward()

            trainable_gradients = [
                p.grad for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]

            if training_mode == "self_supervised":
                trainable_gradients += [
                    p.grad for p in temporal_model.parameters()
                    if p.requires_grad and p.grad is not None
                ]

            if not trainable_gradients:
                raise RuntimeError("No gradients available for training.")
            
            if training_mode == "train_linear":
                # Chỉ cập nhật classifier.
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(temporal_model.parameters()),
                    max_norm=1.0
                )

            model_optimizer.step()
            if training_mode == "self_supervised":
                temporal_optimizer.step()

            batch_size = data.size(0)
            loss_sum += loss.item() * batch_size
            total += batch_size

        train_loss = loss_sum / max(total, 1)
        train_acc = (
            correct / max(total, 1)
            if training_mode != "self_supervised" else 0.0
        )

        valid_loss, valid_acc = evaluate(
            model, temporal_model, valid_dl, device,
            training_mode, configs
        )

        scheduler.step(valid_loss)

        logger.info(
            f"Epoch {epoch}/{configs.num_epoch} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={valid_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={valid_acc:.4f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(
                best_path, epoch, model, temporal_model,
                model_optimizer, temporal_optimizer, best_valid_loss
            )

        save_checkpoint(
            last_path, epoch, model, temporal_model,
            model_optimizer, temporal_optimizer, best_valid_loss
        )

    logger.info(f"Training complete. Best checkpoint: {best_path}")
