# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.init_info = "Welcome! Define your model here!"
        print(self.init_info)

    def forward(self):
        pass
        return

    def training_step(self, batch, loss_compute):
        """
        What to do in the training loop.

        Example:
            x, y = batch
            y_hat = self.forward()
            loss = F.cross_entropy(y_hat, y)
            tensorboard_logs = {'train_loss': loss}
            return {'loss': loss, 'log': tensorboard_logs}
        """
        return

    def validation_step(self, batch):
        """
        What to do in the validation step.

        Example:
            x, y = batch
            y_hat = self.forward(x)
            return {'val_loss': F.cross_entropy(y_hat, y)}
        """
        return

    def validation_end(self, outputs):
        """
        How to aggregate validation_step outputs.

        Example:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            tensorboard_logs = {'val_loss': avg_loss}
            return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
        """
        return

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)
