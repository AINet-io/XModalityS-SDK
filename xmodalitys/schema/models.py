# © Copyright 2023, Liu Yuchen, All rights reserved.

import abc
from enum import IntEnum
from typing import List, Dict, Any

import numpy as np
import torch

from xmodalitys.schema import Modality, Model


class TaskType(IntEnum):
    pass


class ModalResult(abc.ABC):
    @abc.abstractmethod
    def get_tensor(self, modality: Modality | None = None, *args, **kwargs) -> Dict[Modality, torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_result(self, para: str, modality: Modality | None = None, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def modal(self) -> Modality:
        raise NotImplementedError


class ModalityModel(abc.ABC):
    @abc.abstractmethod
    def unimodal(self,
                 *args,
                 texts: List[str] | None = None,
                 images: List[np.ndarray] | None = None,
                 **kwargs) -> ModalResult:
        raise NotImplementedError

    @abc.abstractmethod
    def multimodal(self,
                   *args,
                   texts: torch.Tensor | np.ndarray | None = None,
                   images: torch.Tensor | np.ndarray | None = None,
                   **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def downstream_execute(self, task_name: str, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self,
                task: str,
                *args,
                texts: List[str] | None = None,
                images: List[np.ndarray] | None = None,
                **kwargs):
        raise NotImplemented

    @abc.abstractmethod
    def specify_model(self) -> Model:
        raise NotImplementedError


class ModalityModelLoader(abc.ABC):
    @abc.abstractmethod
    def load_model(self, name: str, version: str = None, modality: Modality = Modality.Unknown,
                   task: str = None, **kwargs) -> ModalityModel:
        raise NotImplementedError
