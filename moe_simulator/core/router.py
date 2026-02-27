"""Token router implementation.

Core routing logic for selecting experts based on scores.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from moe_simulator.core.config import RouterConfig


@dataclass
class TokenRouter:
    """Router for selecting experts based on scoring.

    Implements top-K routing where tokens are dispatched to
    the K experts with highest scores. Provides interface for
    classification-based routing decisions.

    Attributes:
        config: Router configuration parameters.
    """

    config: RouterConfig

    def route(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Route tokens to top-K experts based on scores.

        Args:
            scores: Array of shape (num_tokens, num_experts) containing
                    routing scores for each token-expert pair.

        Returns:
            Tuple of (expert_indices, expert_weights):
            - expert_indices: Array of shape (num_tokens, K) with indices
                              of selected experts
            - expert_weights: Array of shape (num_tokens, K) with routing
                              weights (softmax-normalized scores)
        """
        num_tokens = scores.shape[0]
        K = self.config.K

        # Get top-K indices for each token
        expert_indices = np.argpartition(scores, -K, axis=1)[:, -K:]

        # Sort indices by score (descending) within each token
        rows = np.arange(num_tokens)[:, None]
        sorted_order = np.argsort(-scores[rows, expert_indices], axis=1)
        expert_indices = expert_indices[rows, sorted_order]

        # Get corresponding scores and compute softmax weights
        top_scores = scores[rows, expert_indices]
        expert_weights = self._softmax(top_scores, axis=1)

        return expert_indices, expert_weights

    def route_with_cache_awareness(
        self,
        scores: np.ndarray,
        cached_experts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Route tokens with cache awareness.

        Prefers cached experts when scores are similar to reduce cache misses.

        Args:
            scores: Array of shape (num_tokens, num_experts) containing
                    routing scores.
            cached_experts: Array of expert IDs currently in cache.

        Returns:
            Tuple of (expert_indices, expert_weights, cache_hits):
            - expert_indices: Array of shape (num_tokens, K) with indices
            - expert_weights: Array of shape (num_tokens, K) with weights
            - cache_hits: Boolean array indicating if each selected expert
                          was in cache
        """
        expert_indices, expert_weights = self.route(scores)

        # Check cache hits
        cache_hits = np.isin(expert_indices, cached_experts)

        return expert_indices, expert_weights, cache_hits

    def _softmax(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Compute softmax along specified axis.

        Args:
            x: Input array.
            axis: Axis along which to compute softmax.

        Returns:
            Softmax-normalized array.
        """
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def get_expert_loads(self, expert_indices: np.ndarray) -> Dict[int, float]:
        """Calculate load distribution across experts.

        Args:
            expert_indices: Array of shape (num_tokens, K) with selected
                           expert indices.

        Returns:
            Dictionary mapping expert_id to load (token count).
        """
        unique, counts = np.unique(expert_indices, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
