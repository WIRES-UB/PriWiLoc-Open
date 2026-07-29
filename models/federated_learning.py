"""Code for federated learning model."""

from __future__ import annotations

import torch

from models.aggregation import (
    AggregationStrategy,
    ClientUpdate,
    UniformMeanAggregation,
    WeightedMeanAggregation,
)
from models.model import TrigAOAResNetModel
from utils.config_hydra import Config


class FederatedLearningModel(TrigAOAResNetModel):
    """Add periodic encoder aggregation to the base localization model.

    Args:
        config: Root experiment configuration.
        aggregation_strategy: Optional shared aggregation strategy.
    """

    def __init__(
        self,
        config: Config,
        aggregation_strategy: AggregationStrategy | None = None,
    ):
        """Initialize the base model and aggregation strategy.

        Args:
            config: Root experiment configuration.
            aggregation_strategy: Optional shared aggregation strategy.

        Returns:
            None.
        """
        super().__init__(config)
        self.aggregation_strategy = aggregation_strategy or UniformMeanAggregation(
            reference_client_id=len(self.resnet_encoder_list) - 1,
        )

    def get_resnet_encoder_parameters(self) -> list[dict[str, torch.Tensor]]:
        """Clone the state dictionary of every AP encoder.

        Returns:
            One cloned state dictionary per ResNet encoder.
        """
        return [
            {name: param.clone().detach() for name, param in encoder.state_dict().items()}
            for encoder in self.resnet_encoder_list
        ]

    def set_resnet_encoder_parameters(
        self,
        averaged_parameters: dict[str, torch.Tensor],
        ap_ids: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """Load an aggregated state into selected AP encoders.

        Args:
            averaged_parameters: Aggregated encoder state dictionary.
            ap_ids: AP indexes to update. `None` updates every AP.

        Returns:
            None.
        """
        selected_ap_ids = (
            range(len(self.resnet_encoder_list)) if ap_ids is None else ap_ids
        )
        for ap_id in selected_ap_ids:
            if ap_id < 0 or ap_id >= len(self.resnet_encoder_list):
                raise IndexError(
                    f"AP index {ap_id} is outside [0, {len(self.resnet_encoder_list)})."
                )
            self.resnet_encoder_list[ap_id].load_state_dict(averaged_parameters)

    @staticmethod
    def average_resnet_encoder_parameters(
        client_parameters: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Average encoder states using the legacy last-client buffer policy.

        Args:
            client_parameters: State dictionaries ordered by client ID.

        Returns:
            The uniformly averaged state dictionary.
        """
        updates = [
            ClientUpdate(client_id=client_id, state=state)
            for client_id, state in enumerate(client_parameters)
        ]
        strategy = UniformMeanAggregation(
            reference_client_id=len(updates) - 1,
        )
        return strategy.aggregate(updates).state

    def aggregate_resnet_encoder_parameters(
        self,
        client_parameters: list[dict[str, torch.Tensor]],
        weights: list[float | int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Aggregate encoder states with the configured strategy.

        Args:
            client_parameters: State dictionaries ordered by client ID.
            weights: Optional public weight for each client.

        Returns:
            The canonical aggregate state dictionary.
        """

        resolved_weights = (
            [1] * len(client_parameters)
            if weights is None
            else weights
        )
        if len(resolved_weights) != len(client_parameters):
            raise ValueError("Each client state requires one aggregation weight.")
        updates = [
            ClientUpdate(
                client_id=client_id,
                state=state,
                weight=resolved_weights[client_id],
            )
            for client_id, state in enumerate(client_parameters)
        ]
        return self.aggregation_strategy.aggregate(updates).state

    def get_aggregation_weights(
        self,
        batch,
        n_clients: int,
    ) -> list[float | int]:
        """Resolve explicit per-client weights for the current round.

        Args:
            batch: Current training batch, used for sample-count weights.
            n_clients: Number of client updates in the round.

        Returns:
            One public aggregation weight per client.
        """

        strategy = self.aggregation_strategy
        if not isinstance(strategy, WeightedMeanAggregation):
            return [1] * n_clients
        if strategy.client_weights is not None:
            if len(strategy.client_weights) != n_clients:
                raise ValueError(
                    "Configured client_weights must match the AP count."
                )
            return list(strategy.client_weights)

        features = getattr(batch, "features_2d", None)
        sample_count = (
            int(features.shape[0])
            if isinstance(features, torch.Tensor) and features.ndim > 0
            else 1
        )
        return [sample_count] * n_clients

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Aggregate AP encoder states at configured batch intervals.

        Args:
            outputs: Training-batch output; unused.
            batch: Current training batch.
            batch_idx: Zero-based batch index within the epoch.

        Returns:
            None.
        """
        if (batch_idx + 1) % self.config.model.average_weight_every_n_batches == 0:
            self.print(
                f"epoch: {self.trainer.current_epoch}, batch: {batch_idx}, "
                "averaging client weight"
            )
            client_parameters = self.get_resnet_encoder_parameters()
            weights = self.get_aggregation_weights(
                batch,
                len(client_parameters),
            )
            averaged_parameters = self.aggregate_resnet_encoder_parameters(
                client_parameters,
                weights,
            )
            self.set_resnet_encoder_parameters(averaged_parameters)
