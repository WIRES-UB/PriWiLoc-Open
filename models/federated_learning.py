"""Code for federated learning model."""

import torch

from models.model import TrigAOAResNetModel
from utils.config_hydra import Config

BATCH_NORM_STATISTICS = ("running_mean", "running_var", "num_batches_tracked")

class FederatedLearningModel(TrigAOAResNetModel):
    """
    A class for the Federated Learning model that extends the TrigAOAResNetModel.
    It implements federated learning specific functionalities.
    """

    def __init__(self, config: Config):
        super().__init__(config)

    def get_resnet_encoder_parameters(self) -> list[dict[str, torch.Tensor]]:
        """
        Extract parameters from self.resnet_encoder_list.

        Returns:
            A list of dictionaries, where each dictionary contains the parameter names and their corresponding tensors
            for a ResNetEncoder.
        """
        return [
            {name: param.clone().detach() for name, param in encoder.state_dict().items()}
            for encoder in self.resnet_encoder_list
        ]

    def set_resnet_encoder_parameters(self, averaged_parameters: dict[str, torch.Tensor]) -> None:
        """
        Set the parameters of self.resnet_encoder_list with averaged global parameters.

        Args:
            averaged_parameters: A single state_dict containing the averaged global parameters for all encoders.
        """
        for encoder in self.resnet_encoder_list:
            encoder.load_state_dict(averaged_parameters)

    @staticmethod
    def average_resnet_encoder_parameters(
        client_parameters: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """
        Average the parameters of ResNet encoders across multiple clients.

        Args:
            client_parameters: A list of state_dicts, where each state_dict corresponds to the
                parameters of a ResNetEncoder from a client.

        Returns:
            A single state_dict containing the averaged parameters for the ResNetEncoder.
        """
        num_clients = len(client_parameters)

        # Initialize an empty state_dict for averaging
        averaged_parameters = {name: torch.zeros_like(param) for name, param in client_parameters[0].items()}

        # Sum parameters from all clients
        for client_param in client_parameters:
            for param_name, param in client_param.items():
                # BatchNorm statistics should not be averaged
                if any(batch_norm_stat in param_name for batch_norm_stat in BATCH_NORM_STATISTICS):
                    averaged_parameters[param_name] = param.clone()
                else:
                    averaged_parameters[param_name] += param

        # Average the parameters. Skip BatchNorm statistics.
        for param_name in averaged_parameters:
            if not any(batch_norm_stat in param_name for batch_norm_stat in BATCH_NORM_STATISTICS):
                averaged_parameters[param_name] /= num_clients

        return averaged_parameters

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Averages the client weight after every n batches.

        Args:
            outputs: not used, but required by the PyTorch Lightning API.
            batch: not used, but required by the PyTorch Lightning API.
            batch_idx: The index of the batch that just ended.
        """
        if (batch_idx + 1) % self.config.model.average_weight_every_n_batches == 0:
            self.print(f"epoch: {self.trainer.current_epoch}, batch: {batch_idx}, averaging client weight")
            client_parameters = self.get_resnet_encoder_parameters()
            averaged_parameters = self.average_resnet_encoder_parameters(client_parameters)
            self.set_resnet_encoder_parameters(averaged_parameters)
