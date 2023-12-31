﻿# © Copyright 2023, Liu Yuchen, All rights reserved.

import time
import uuid
from enum import IntEnum
from typing import List, Dict, Any

import numpy as np
from pydantic import BaseModel, Field

VERSION: str = "XModalityS.Entity.0.0.1"
VERSIONS: Dict[str, str] = {
    "0.0.1": VERSION,
    "latest": VERSION,
}


class Modality(IntEnum):
    Unknown = 0
    Language = 1
    Vision = 1 << 1


class RawResourceKind(IntEnum):
    Custom = 0
    Vector = 1
    Text = 2
    Image = 3

    @classmethod
    def from_str(cls, kind: str):
        if kind == "text":
            return cls.Text

        if kind == "image":
            return cls.Image

        if kind == "vector":
            return cls.Vector

        return cls.Custom


class Metadata(BaseModel):
    name: str = Field(title="entity data unique name.")
    l2norm: float = Field(default=None, title="vector Euclidean distance, equalling to vector^2")
    creationTimestamp: float = Field(default_factory=time.time, title="data creation timestamp.")


class Model(BaseModel):
    name: str = Field(title="model name")
    revision: str = Field(default=None, title="model revision.")
    task: str | None = Field(default=None, title="model processes task type")


class ModelCard(BaseModel):
    name: str = Field(title="model name")
    revision: str = Field(default=None, title="model revision.")
    modality: int = Field(default=Modality.Unknown, title="model capable modality.")
    capabilities: List[str] | None = Field(default=Modality.Unknown, title="model capable tasks.")
    labels: Dict[str, str] | None = Field(default=None, title="model card labels.")


class RawStore(BaseModel):
    name: str | None = Field(default=None, title="raw store name.")
    kind: str | None = Field(default=None, title="raw store type, url/s3.")
    bucket: str | None = Field(default=None, title="store bucket.")
    endpoints: str | None = Field(default=None, title="raw store endpoints.")
    path: str | None = Field(default=None, title="raw resource stored information.")
    labels: Dict[str, str] | None = Field(default=None, title="store labels.")


class SubSlice(BaseModel):
    start: int | None = Field(default=None, title="slice start number in dataset.")
    end: int | None = Field(default=None, title="slice end number in dataset.")
    series: List[str] | List[np.ndarray] | None = Field(default=None, title="series slice of dataset.")

    class Config:
        arbitrary_types_allowed = True


class Source(BaseModel):
    name: str | None = Field(default=None, title="raw resource name, the same of entity name by default.")
    kind: RawResourceKind | None = Field(default=None, title="raw resource kind, text/image/audio/video/tensor or" +
                                                             " other custom source kind.")
    raw: RawStore = Field(default=None, title="raw store information")
    aggregation: SubSlice = Field(default=None, title="aggregation of tensors if kind come from tensor.")


class Cluster(BaseModel): ...


class Dataset(BaseModel):
    name: str | None = Field(default=None, title="dataset name.")
    subslice: SubSlice = Field(default=None, title="dataset sub-slice.")
    cluster: Cluster | None = Field(default=None, title="cluster information from dataset slice.")
    labels: Dict[str, Any] | None = Field(default=None, title="dataset custom labels.")


class Entity(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), title="entity main key.")
    metadata: Metadata = Field(default=Metadata(name=""), title="entity basic information.")
    model: Model = Field(default=Model(name="default", revision="default"), title= \
        "model information, schema information -> Model.")
    # TODO: rename vector to unit.
    vector: List[float] | np.ndarray = Field(title="entity vector generated by transformer-style model.")
    # unit: List[float] | np.ndarray | None= Field(default=None, title="a unit vector that can calculate similarity.")
    modality: Modality = Field(default=Modality.Unknown, title="entity modality.")
    dataset: Dataset = Field(default=Dataset(), title="entity logical information, dataset which entity belongs to.")
    source: Source = Field(default=Source(), title="entity raw source information which generates entity.")
    labels: Dict[str, str] = Field(default={}, title="custom labels which marks entity.")

    class Config:
        arbitrary_types_allowed = True


class EntityGenerator:
    def __init__(self, model_name: str, model_revision: str, task: str, modality: Modality = Modality.Unknown):
        self._model = Model(name=model_name, revision=model_revision, task=task)
        self._modality = modality

    @property
    def model(self) -> Model:
        return self._model

    @property
    def modality(self) -> Modality:
        return self._modality

    def generate_entity(self,
                        name: str,
                        vector: List[float] | np.ndarray,
                        modality: Modality = None,
                        dataset: Dataset | None = Dataset(),
                        id: str = None,
                        source: Source | None = Source(),
                        labels: {str: str} = {},
                        metadata: Metadata | None = None,
                        **kwargs) -> Entity:
        dataset = Dataset() if dataset is None else dataset
        source = Source() if source is None else source
        id = id if id is not None else str(uuid.uuid4())
        metadata = metadata if metadata is not None else Metadata(name=name)
        return Entity(metadata=metadata,
                      model=self.model,
                      vector=vector,
                      uuid=id,
                      dataset=dataset,
                      modality=modality if modality is not None else self.modality,
                      source=source,
                      labels=labels)
