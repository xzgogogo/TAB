import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import optim

from ts_benchmark.baselines.self_impl.CrossAD.models.CrossAD_model import (
    Basic_CrossAD,
)
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.baselines.utils import train_val_split
from .utils.tools import EarlyStopping, _adjust_learning_rate


DEFAULT_CrossAD_BASED_HYPER_PARAMS = {
    "seq_len": 192,
    "patch_len": 8,
    "ms_kernels": [24, 12, 6],
    "ms_method": "average_pooling",
    "topk": 10,
    "n_query": 5,
    "query_len": 5,
    "bank_size": 32,
    "decay": 0.9,
    "epsilon": 1e-5,
    "e_layers": 1,
    "d_layers": 1,
    "m_layers": 1,
    "n_heads": 8,
    "attn_dropout": 0,
    "proj_dropout": 0,
    "d_model": 256,
    "d_ff": None,
    "ff_dropout": 0,
    "norm": "layernorm",
    "activation": "gelu",
    "train_epochs": 20,
    "batch_size": 128,
    "optim": "adam",
    "learning_rate": 1e-4,
    "lradj": "type1",
    "patience": 3,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


class CrossADConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_CrossAD_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class CrossAD:
    def __init__(self, **kwargs):
        super(CrossAD, self).__init__()
        self.config = CrossADConfig(**kwargs)
        self.scaler = StandardScaler()
        self.model_name = "CrossAD"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {}

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

    def detect_validate(self, valid_data_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (input, _) in enumerate(valid_data_loader):
                input = input.float().to(self.device)
                ms_loss, q_latent_distance = self.model(input, None, None, None)
                loss = ms_loss
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Train the model.

        :param train_data: Time series data used for training.
        """

        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.model = Basic_CrossAD(self.config)
        self.model.to(self.device)

        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )

        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)

        train_steps = len(self.train_data_loader)

        if config.optim == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif config.optim == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        if config.lradj == "1cycle":
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                      steps_per_epoch=train_steps,
                                                      pct_start=self.config.pct_start,
                                                      epochs=self.config.train_epochs,
                                                      max_lr=self.config.learning_rate
                                                      )

        time_now = time.time()

        for epoch in range(self.config.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            self.model.train()

            for i, (input, target) in enumerate(self.train_data_loader):
                iter_count += 1
                self.optimizer.zero_grad()

                input = input.float().to(self.device)
                ms_loss, q_latent_distance = self.model(input, None, None, None)
                loss = ms_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

                if self.config.lradj == '1cycle':
                    _adjust_learning_rate(self.optimizer, epoch + 1, self.config, verbose=False,
                                               scheduler=scheduler)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.detect_validate(self.valid_data_loader)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps,
                                                                                           train_loss, vali_loss))

            self.early_stopping(vali_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            if self.config.lradj != "1cycle":
                _adjust_learning_rate(self.optimizer, epoch + 1, self.config)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        self.model.to(self.device)
        self.model.eval()
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.thre_loader):
                batch_x = batch_x.float().to(self.device)
                score, q_latent_distance = self.model.infer(batch_x, None, None, None)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

                print(f"\t testing loss: {score[0, :5]}")

        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # test_energy = np.array(attens_energy)

            attens_energy = np.concatenate(attens_energy, axis=0)  # nb x t x c
            attens_energy = attens_energy.reshape(-1, attens_energy.shape[-1])  # nb*t x c
            test_energy = np.array(attens_energy)

        test_energy = np.mean(test_energy, axis=-1)

        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.test_data_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
        )

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        attens_energy = []

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.train_data_loader):
                batch_x = batch_x.float().to(self.device)
                score, q_latent_distance = self.model.infer(batch_x, None, None, None)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # train_energy = np.array(attens_energy)

            attens_energy = np.concatenate(attens_energy, axis=0)  # nb x t x c
            attens_energy = attens_energy.reshape(-1, attens_energy.shape[-1])  # nb*t x c
            train_energy = np.array(attens_energy)

        train_energy = np.mean(train_energy, axis=-1)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.test_data_loader):
                batch_x = batch_x.float().to(self.device)
                score, q_latent_distance = self.model.infer(batch_x, None, None, None)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # test_energy = np.array(attens_energy)
        # combined_energy = np.concatenate([train_energy, test_energy], axis=0)

            attens_energy = np.concatenate(attens_energy, axis=0)  # nb x t x c
            attens_energy = attens_energy.reshape(-1, attens_energy.shape[-1])  # nb*t x c
            test_energy = np.array(attens_energy)

        test_energy = np.mean(test_energy, axis=-1)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.thre_loader):
                batch_x = batch_x.float().to(self.device)
                score, q_latent_distance = self.model.infer(batch_x, None, None, None)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

                print(f"\t testing loss: {score[0, :5]}")

        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # test_energy = np.array(attens_energy)

            attens_energy = np.concatenate(attens_energy, axis=0)  # nb x t x c
            attens_energy = attens_energy.reshape(-1, attens_energy.shape[-1])  # nb*t x c
            test_energy = np.array(attens_energy)

        test_energy = np.mean(test_energy, axis=-1)

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy
