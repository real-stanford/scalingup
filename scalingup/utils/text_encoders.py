from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple
import typing

from transformers import CLIPModel, CLIPProcessor

from scalingup.algo.algo import TrainableScalingUpAlgo
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ClipLanguageConditioned(torch.nn.Module):
    clip_model: Optional[CLIPModel] = None
    clip_processor: Optional[CLIPProcessor] = None

    def __init__(
        self,
        precache_text_descs: Optional[List[str]] = None,
        use_one_hot: bool = False,
        use_last_hidden_state: bool = False,
        max_descs: int = 32,
        text_dim: int = 512,
    ):
        super().__init__()
        self.lang_embs_cache = torch.nn.ParameterDict({})
        self.use_one_hot = use_one_hot
        self.use_last_hidden_state = use_last_hidden_state
        self.max_descs = max_descs
        self.text_dim = text_dim
        if self.use_one_hot:
            logging.info("Using one-hot text encoding")
        # pre-cache some text features
        if precache_text_descs is not None:
            for text in precache_text_descs:
                self.get_text_feature(text)

    @classmethod
    def get_clip_model_and_processor(
        cls, model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu"
    ) -> Tuple[CLIPModel, CLIPProcessor]:
        if cls.clip_model is None:
            logging.info(f"Loading CLIP Model ({model_id})")
            cls.clip_model = CLIPModel.from_pretrained(model_id).to(device)  # type: ignore
            for param in cls.clip_model.parameters():
                param.requires_grad = False
            cls.clip_model.eval()
            cls.clip_processor = CLIPProcessor.from_pretrained(model_id)
        assert cls.clip_model is not None
        assert cls.clip_processor is not None
        return cls.clip_model, cls.clip_processor

    def get_text_feature(self, text: str) -> torch.Tensor:
        if text not in self.lang_embs_cache:
            logging.info(f"Getting text features for {text!r}")
            if self.use_one_hot:
                if len(self.lang_embs_cache) >= self.max_descs:
                    raise RuntimeError(
                        f"Cannot add more descriptions, max_descs={self.max_descs}"
                    )
                self.lang_embs_cache[text] = torch.nn.Parameter(
                    torch.nn.functional.one_hot(
                        torch.tensor([len(self.lang_embs_cache)]),
                        num_classes=self.text_dim,
                    ).squeeze(),
                    requires_grad=False,
                )
            else:
                model, processor = self.get_clip_model_and_processor()
                with torch.no_grad():
                    inputs = {
                        k: v.to(model.device)
                        for k, v in processor(
                            text=text,
                            return_tensors="pt",
                            padding="max_length",
                        ).items()
                    }
                    output = model.text_model(
                        return_dict=True,
                        output_hidden_states=True,
                        output_attentions=False,
                        **inputs,
                    )
                    if self.use_last_hidden_state:
                        self.lang_embs_cache[text] = torch.nn.Parameter(
                            output.last_hidden_state.squeeze(),
                            requires_grad=False,
                        )
                    else:
                        self.lang_embs_cache[text] = torch.nn.Parameter(
                            output.pooler_output.squeeze(),
                            requires_grad=False,
                        )
        return self.lang_embs_cache[text]  # type: ignore
