# © Copyright 2023, Liu Yuchen, All rights reserved.

import abc
from typing import Union, List, Dict, Optional, Any

import numpy as np
from pydantic import BaseModel, Field

from xmodalitys.schema.entity import Entity
from xmodalitys.schema.error import XModalitySError


class ModalityStoreError(XModalitySError): ...


class ModalityStoreInformation(BaseModel):
    collection_name: str = Field(title="collection name.", allow_mutation=False)

    class Config:
        validate_assignment = True


class DeleteResult(BaseModel):
    primary_keys: List[str] = Field(default=[], title="delete items primary keys.", allow_mutation=False)
    delete_count: int = Field(default=0, title="deleted item result count.", allow_mutation=False)
    timestamp: int = Field(default=0, title="delting executed timestamp.", allow_mutation=False)
    succ_count: int = Field(default=0, title="successful deleted items count.", allow_mutation=False)
    err_count: int = Field(default=0, title="failed deleted items count.", allow_mutation=False)

    class Config:
        validate_assignment = True


class InsertResult(BaseModel):
    primary_keys: List[str] | str = Field(default=[], title="delete items primary keys.", allow_mutation=False)
    insert_count: int = Field(default=0, title="insert item result count.", allow_mutation=False)
    upsert_count: int = Field(default=0, title="upsert item result count.", allow_mutation=False)
    timestamp: int = Field(default=0, title="delting executed timestamp.", allow_mutation=False)
    succ_count: int = Field(default=0, title="successful deleted items count.", allow_mutation=False)
    err_count: int = Field(default=0, title="failed deleted items count.", allow_mutation=False)

    class Config:
        validate_assignment = True


class Hit(BaseModel):
    entity: Entity = Field(title="raw hit entity.", allow_mutation=False)
    id: Any = Field(default=None, title="store id.", allow_mutation=False)
    distance: float | None = Field(default=None, title="search distance from target ranging", allow_mutation=False)
    similarity: float | None = Field(default=None, title="search similarity from target ranging from 0.0 to 1.0",
                                     allow_mutation=False)

    class Config:
        validate_assignment = True

    def __init__(self, hit: Dict, id: int | None, distance: float | None, similarity: float | None):
        """Hit init with Entity schema like dict.

        Args:
            hit (Dict): hit is a search hit result. see Entity schema.
        """
        keys = list(Entity(vector=[]).__dict__.keys())
        entity_dict = {}
        for key in keys:
            value = hit.get(key, None)
            if value is None:
                continue
            entity_dict[key] = value
        super(Hit, self).__init__(entity=Entity(**entity_dict), id=id, distance=distance, similarity=similarity)

    def __str__(self) -> str:
        """
        Return the information of hit record.

        :return str:
            The information of hit record.
        """
        return str(self.dict())

    __repr__ = __str__


class SearchResult(BaseModel):
    hits: List[Hit] = Field(title="search hit result.")

    def __init__(self, hits: List[Hit]):
        """
        Construct a Hits object from response.
        """
        super(SearchResult, self).__init__(hits=hits)

    def __iter__(self):
        """
        Iterate the Hits object. Every iteration returns a Hit which represent a record
        corresponding to the query.
        """
        return self

    def __next__(self):
        """
        Iterate the Hits object. Every iteration returns a Hit which represent a record
        corresponding to the query.
        """
        return next(iter(self.hits))

    def __len__(self) -> int:
        """
        Return the number of hit record.

        :return int:
            The number of hit record.
        """
        return self.hits.__len__()

    def __str__(self) -> str:
        return str(self.hits)

    @property
    def ids(self) -> List[int]:
        """
        Return the ids of all hit record.

        :return list[int]:
            The ids of all hit record.
        """
        return [hit.id for hit in self.hits]

    @property
    def distances(self) -> List[float]:
        """
        Return the distances of all hit record.

        :return list[float]:
            The distances of all hit record.
        """
        return [hit.distance for hit in self.hits]

    @property
    def similarities(self) -> List[float]:
        """_summary_

        Returns:
            list[float]: _description_
        """
        return [hit.similarity for hit in self.hits]

    @property
    def entities(self) -> List[Entity]:
        return [hit.entity for hit in self.hits]


class ModalityStore(abc.ABC):
    @abc.abstractmethod
    def insert(self,
               data: List[Entity] | Entity,
               *args,
               **kwargs) -> InsertResult:
        raise NotImplementedError

    @abc.abstractmethod
    def upsert(self,
               data: List[Entity] | Entity,
               *args,
               **kwargs) -> InsertResult:
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, *args, expr: str | None = None, **kwargs) -> DeleteResult:
        raise NotImplementedError

    @abc.abstractmethod
    def search(self,
               vectors: np.ndarray,
               *args,
               offset: int = 0,
               limit: int = 10,
               expr: str | None = None,
               output_fields: List[str] = None,
               **kwargs) -> Union[SearchResult, List[SearchResult]]:
        raise NotImplementedError

    @abc.abstractmethod
    def query(self,
              *args,
              expr: str | None = None,
              output_fields: Optional[List[str]] = None,
              offset: int = 0,
              limit: int = 10,
              **kwargs) -> SearchResult:
        raise NotImplementedError

    @abc.abstractmethod
    def specify_store(self) -> ModalityStoreInformation:
        raise NotImplementedError
