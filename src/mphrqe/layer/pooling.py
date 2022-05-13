"""Graph pooling operations."""
from abc import abstractmethod
from typing import Optional

import torch
from class_resolver import Resolver
from torch import nn
from torch_scatter import scatter_add

from gqs.mapping import get_entity_mapper, EntityMapper
from ..typing import FloatTensor

__all__ = [
    "graph_pooling_resolver",
    "GraphPooling",
]


class GraphPooling(nn.Module):
    """A module for graph pooling."""

    @abstractmethod
    def forward(
        self,
        x_e: torch.FloatTensor,
        graph_ids: torch.LongTensor,
        entity_ids: Optional[torch.LongTensor],
    ) -> FloatTensor:
        """
        Obtain graph representations by aggregating node representations.

        :param x_e: shape: (num_nodes, dim)
            The node representations.
        :param graph_ids: shape: (num_nodes,)
            The graph ID for each node.
        :param entity_ids: shape: (num_nodes,)
            The global entity ID for each node.

        :return: shape: (num_graphs, dim)
            The graph representations.
        """
        raise NotImplementedError


class SumGraphPooling(GraphPooling):
    """Aggregation by sum."""

    def forward(
        self,
        x_e: torch.FloatTensor,
        graph_ids: torch.LongTensor,
        entity_ids: Optional[torch.LongTensor] = None,
    ) -> FloatTensor:  # noqa: D102
        return scatter_add(x_e, index=graph_ids, dim=0)


class TargetPooling(GraphPooling):
    """Aggregation by sum."""
    entmap: EntityMapper

    def set_entmap(self, entmap: EntityMapper):
        self.entmap = entmap

    def forward(
        self,
        x_e: torch.FloatTensor,
        graph_ids: torch.LongTensor,
        entity_ids: Optional[torch.LongTensor] = None,
    ) -> FloatTensor:  # noqa: D102
        """
        graph_ids: binary mask
        """
        assert entity_ids is not None
        mask = entity_ids == self.entmap.number_of_entities_and_reified_relations_without_vars_and_targets() + 1
        assert mask.sum() == graph_ids.unique().shape[0], "There should be exactly one target node per graph."
        return x_e[mask]


graph_pooling_resolver: Resolver[GraphPooling] = Resolver.from_subclasses(
    base=GraphPooling,  # type: ignore
    default=SumGraphPooling,
)
