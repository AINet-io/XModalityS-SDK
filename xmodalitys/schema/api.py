# © Copyright 2023, Liu Yuchen, All rights reserved.

from typing import List, Dict, Any

import numpy as np
import pandas as pd
import ray
from pydantic import BaseModel, Field
from ray.data import Schema

from xmodalitys.schema.entity import Source, Modality, RawStore, SubSlice, Model, RawResourceKind, ModelCard
from xmodalitys.schema.error import XModalitySInternalError, XModalitySCode, XModalitySExternalError
from xmodalitys.schema.storage import InsertResult, SearchResult


class RegisterModelSpawner(BaseModel):
    name: str | None = Field(default=None, title="register model spawner deployment name")
    app_name: str | None = Field(default=None, title="register model spawner deployment app name")
    model: ModelCard = Field(title="modality model spawner information")


class RegistryGeneralResponse(BaseModel):
    entities: List[Any] | None = Field(default=None, title="modality model information")
    error: str | None = Field(default=None, title="registry error information")


class ModalRawResult(BaseModel):
    tensor: List[Dict[str, Any]] = Field(title="raw modality tensor result, converted to list")
    result: Any = Field(default=None, title="raw modality result")

    class Config:
        arbitrary_types_allowed = True


class ModelhubResponse(BaseModel):
    modal_result: ModalRawResult | None = Field(default=None, title="modalling result.")
    error: Any | None = Field(default=None, title="model hub error response, if no error, should be None")

    def __init__(self, modal_result: ModalRawResult | None = None, error: BaseException | None = None):
        if isinstance(error, XModalitySInternalError):
            raise XModalitySInternalError(f"internal error {e} should not expose")
        super(ModelhubResponse, self).__init__(modal_result=modal_result, error=error)


class DatahubResponse(BaseModel):
    dataset: Any | None = Field(default=None, title="datahub response.")
    error: Any | None = Field(default=None, title="model hub error response, if no error, should be None")

    def __init__(self, dataset: Any | None = None, error: BaseException | None = None):
        if isinstance(error, XModalitySInternalError):
            raise XModalitySInternalError(f"internal error {e} should not expose")
        super(DatahubResponse, self).__init__(dataset=dataset, error=error)


class RegisterDataLoader(BaseModel):
    name: str | None = Field(default=None, title="register data loader spawner deployment name")
    app_name: str | None = Field(default=None, title="register data loader spawner deployment app name")
    raw: RawStore | None = Field(default=None, title="register data loader spawner deployment app name")


class ModelInformation(BaseModel):
    name: str = Field(title="model name")
    revision: str | None = Field(default=None, title="model revision.")


class Vector(BaseModel):
    name: str
    data: List[float] | np.ndarray
    modality: Modality = Modality.Unknown
    source: Source | None = Field(default=None, title="generate vector source.")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_dict(cls, dictionary: dict):
        vector = cls(name="")
        for key, value in dictionary.items():
            setattr(vector, key, value)
        return vector


class DataSchemaError(XModalitySExternalError): ...


class DataSource(BaseModel):
    name: str | None = Field(default=None, title="data source name, if is None, set by RawStore.path")
    raw: RawStore = Field(title="uploading raw store information.")
    kind: RawResourceKind | None = Field(default=None, title="uploading raw resource kind.")
    subslice: SubSlice | None = Field(default=None, title="sub-slice information of raw resource.")
    labels: Dict[str, Any] | None = Field(default=None, title="data source custom labels.")

    def __init__(self, name: str | None = None, raw: RawStore | None = None, kind: str | RawResourceKind | None = None,
                 subslice: SubSlice | None = None, labels: Dict[str, Any] | None = None):
        if isinstance(kind, str):
            kind = RawResourceKind.from_str(kind)
        super(DataSource, self).__init__(name=name, raw=raw, kind=kind, subslice=subslice, labels=labels)


class Data(BaseModel):
    name: str | None = Field(default=None, title="data unique name in vector store")
    data: Any = Field(title="data")
    raw: RawStore | None = Field(default=None, title="data raw store information.")


class Dataset(BaseModel):
    name: str | None = Field(default=None, title="dataset name")
    dataset: List[Data] | pd.DataFrame | ray.data.Dataset = Field(title="dataset")
    raw: RawStore | None = Field(default=None, title="images raw information.")
    labels: Dict[str, Any] | None = Field(default=None, title="dataset labels.")

    class Config:
        arbitrary_types_allowed = True


class ImageFrame(BaseModel):
    name: str = Field(title="image unique name in vector store")
    frame: np.ndarray | List[List[int]] = Field(title="image frame")
    raw: RawStore | None = Field(default=None, title="images raw information.")

    class Config:
        arbitrary_types_allowed = True


# TODO: using ray.put() store data, reserving ref on self.frames
class ImageFrameDataset(BaseModel):
    name: str | None = Field(default=None, title="dataset name.")
    frames: List[ImageFrame] | pd.DataFrame | ray.data.Dataset = Field(title="image frame dataset.")
    raw: RawStore | None = Field(default=None, title="images raw information.")
    labels: Dict[str, Any] | None = Field(default=None, title="dataset labels.")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self,
                 frames: List[ImageFrame] | pd.DataFrame | ray.data.Dataset,
                 name: str | None = None,
                 raw: RawStore | None = None,
                 labels: Dict[str, Any] | None = None):
        valid_keys = {"name", "frame"}

        if isinstance(frames, pd.DataFrame):
            key_set = set(frames.keys())
            if not key_set.issuperset(valid_keys):
                raise DataSchemaError(error=f"input dataframe is not valid, should contains {valid_keys}")

        elif isinstance(frames, ray.data.Dataset):
            schema: Schema = frames.schema()
            key_set = set(schema.names)
            if not key_set.issuperset(valid_keys):
                raise DataSchemaError(error=f"input dataframe is not valid, should contains {valid_keys}")

        super(ImageFrameDataset, self).__init__(name=name, frames=frames, labels=labels)

    def get_images(self) -> List[np.ndarray]:
        if isinstance(self.frames, list):
            data = []
            for f in self.frames:
                data.append(f.frame)
            return data

        elif isinstance(self.frames, pd.DataFrame):
            data = self.frames["frame"].to_list()
            return data
        elif isinstance(self.frames, ray.data.Dataset):
            pass

    def get_names(self) -> List[str]:
        if isinstance(self.frames, list):
            names = []
            for f in self.frames:
                names.append(f.name)
            return names

        elif isinstance(self.frames, pd.DataFrame):
            names = self.frames["name"].to_list()
            return names


class Search(BaseModel):
    model_name: str
    vectors: List[List[float]] | List[float] | np.ndarray
    limit: int
    offset: int = 0
    expr: str | None = None
    model_revision: str = "default"
    output_fields: List[str] | None = None

    class Config:
        arbitrary_types_allowed = True


class Insert(BaseModel):
    model_name: str
    vectors: List[List[float]] | List[float] | np.ndarray = Field(
        title="vectors should be a list of dict or just a dict, which includes key \"name\" specifies vector " + \
              "uuid and \"data\" vector itself.",
    )
    model_revision: str = "default"
    default_modality: Modality = Field(
        default=Modality.Unknown,
        title="vector default modality, if vector had set no modality",
    )
    model_modality: Modality = Field(
        default=Modality.Unknown,
        title="model capable modality, if a languale model, it should be 1. If a vision model, " + \
              "it should be 2. If a language and vision model, it should be 1|2 = 3"
    )
    data_source: DataSource = Field(default=None, title="insert data source information.")

    class Config:
        arbitrary_types_allowed = True


class Delete(BaseModel):
    model_name: str
    expr: str
    model_revision: str = "default"


class Query(BaseModel):
    model_name: str
    limit: int
    offset: int = 0
    model_revision: str = "default"
    output_fields: List[str] = None


class InsertDatasets(BaseModel):
    data_source: List[DataSource] | DataSource = Field(title="insert data source.")
    model: ModelInformation | None = Field(default=None, title="inference insert images model")
    task: str | None = Field(default=None, title="task that model should execute.")
    collection: str | None = Field(default=None, title="insert collection name.")
    timeout: float | None = Field(default=None, title="insert timeout, if timeout is empty, wait until done.")


class InsertDatasetResult(BaseModel):
    name: str = Field(title="dataset name")
    code: XModalitySCode = Field(default=XModalitySCode.Success, title="insert images result code")
    error: str | None = Field(default=None, title="insert dataset error")
    result: InsertResult | None = Field(default=None, title="insert result")


class MultimodalRetrieval(BaseModel):
    text: str | None = Field(default=None, title="retrieve matched vector information that described by text.")
    image: List[List[List[int]]] | None = Field(default=None,
                                                title="retrieve matched vector information that described by text.")
    task: str | None = Field(default=None, title="retrieval task name.")
    expr: str | None = Field(default=None, title="query expression that filter certain condition.")
    model: ModelInformation | None = Field(default=None, title="modality model name and revision")
    collection: str | None = Field(default=None, title="name of collection that text/image expects retrieval from.")
    start: int = Field(default=0, title="search result start.")
    count: int = Field(default=10, title="search result count.")
    projector: List[str] | None = Field(default=None, title="search result projector.")


class MultimodalRetrievalResult(BaseModel):
    error: str | None = Field(default=None, title="retrieval error information.")
    result: List[SearchResult] | SearchResult | None = Field(default=None, title="search result.")
