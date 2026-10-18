import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss



def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    best_valid_loss = float("inf")

    saved_models_dir = os.path.join(
        experiment_log_dir, "saved_models"
    )
    os.makedirs(saved_models_dir, exist_ok=True)

    for epoch in range(1, config.num_epoch + 1):
        train_loss, train_acc = model_train(
            model, temporal_contr_model,
            model_optimizer, temp_cont_optimizer,
            criterion, train_dl, config, device,
            training_mode
        )

        valid_loss, valid_acc, _, _ = model_evaluate(
            model, temporal_contr_model,
            valid_dl, device, training_mode
        )

        if training_mode != "self_supervised":
            scheduler.step(valid_loss)

        logger.debug(
            f"\nEpoch : {epoch}\n"
            f"Train Loss     : {float(train_loss):.4f}"
            f"\t | Train Accuracy : {float(train_acc):.4f}\n"
            f"Valid Loss     : {float(valid_loss):.4f}"
            f"\t | Valid Accuracy : {float(valid_acc):.4f}"
        )

        # Lưu mô hình có validation loss thấp nhất
        current_valid_loss = float(valid_loss)

        if current_valid_loss < best_valid_loss:
            best_valid_loss = current_valid_loss

            checkpoint = {
                "epoch": epoch,
                "best_valid_loss": best_valid_loss,
                "model_state_dict": model.state_dict(),
                "temporal_contr_model_state_dict":
                    temporal_contr_model.state_dict(),
                "model_optimizer_state_dict":
                    model_optimizer.state_dict(),
            }

            torch.save(
                checkpoint,
                os.path.join(saved_models_dir, "ckp_best.pt")
            )

            logger.debug(
                f"Saved best model at epoch {epoch}, "
                f"validation loss = {best_valid_loss:.4f}"
            )

    # Vẫn giữ checkpoint cuối để tiện đối chiếu
    checkpoint_last = {
        "model_state_dict": model.state_dict(),
        "temporal_contr_model_state_dict":
            temporal_contr_model.state_dict(),
    }

    torch.save(
        checkpoint_last,
        os.path.join(saved_models_dir, "ckp_last.pt")
    )

    logger.debug(
        f"Best validation loss: {best_valid_loss:.4f}"
    )

    logger.debug(
        "\n################## Training is Done! #########################"
    )

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()
    if training_mode == "train_linear":
        # Giữ encoder ở chế độ eval:
        # BatchNorm không cập nhật running statistics,
        # Dropout không ngẫu nhiên tắt các đặc trưng.
        for module in model.modules():
            if isinstance(
                module,
                (nn.BatchNorm1d, nn.Dropout)
            ):
                module.eval()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1 
            zjs = temp_cont_lstm_feat2 

        else:
            output = model(data)

        # compute loss
        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
            
        else: # supervised training or fine tuining
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs
