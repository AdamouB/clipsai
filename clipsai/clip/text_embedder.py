"""Embed text using the Roberta model."""

from __future__ import annotations

# 3rd party imports
import torch

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - allow import without package
    SentenceTransformer = None  # type: ignore


class TextEmbedder:
    """
    A class for embedding text using the Roberta model.
    """

    def __init__(self) -> None:
        """Initialise the text embedding model if dependencies exist."""
        if SentenceTransformer is None:
            raise ModuleNotFoundError(
                "sentence_transformers is required to use TextEmbedder"
            )
        self.__model = SentenceTransformer("all-roberta-large-v1")

    def embed_sentences(self, sentences: list) -> torch.Tensor:
        """
        Creates embeddings for each sentence in sentences

        Parameters
        ----------
        sentences: list
            a list of N sentences

        Returns
        -------
        - sentence_embeddings: torch.tensor
            a tensor of N x E where n is a sentence and e
            is an embedding for that sentence
        """
        if torch is None:
            raise ModuleNotFoundError("torch is required to embed sentences")
        return torch.tensor(self.__model.encode(sentences))
