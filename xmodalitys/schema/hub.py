# © Copyright 2023, Liu Yuchen, All rights reserved.

import abc
import inspect
from typing import Any, Dict, List, Tuple
import uuid
import functools

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

from xmodalitys.schema.api import DataSource, ImageFrameDataset, Dataset, RegistryGeneralResponse
from xmodalitys.schema.api import RegisterDataLoader, RegisterModelSpawner
from xmodalitys.schema.entity import RawStore, ModelCard
from xmodalitys.schema.error import XModalitySExternalError, XModalitySInternalError
from xmodalitys.schema.models import ModalityModel

DATAHUB_REGISTRY = "DatahubRegistry"
DATAHUB_APP = "DatahubRegistry"
MODELHUB_REGISTRY = "ModelhubRegistry"
MODELHUB_APP = "ModelhubRegistry"



def _check_resource_exists(app_name:str, name:str, query: Any, registry: DeploymentHandle) -> Any:
    try:
        _ = serve.get_deployment_handle(deployment_name=name, app_name=app_name)
    except (serve.exceptions.RayServeException, KeyError):
        return None

    response: RegistryGeneralResponse = registry.get.remote(query).result()

    if response.error is not None:
        raise XModalitySInternalError(error=f"get entity error: {response.error}")
    
    if len(response.entities) == 1:
        return response.entities[0]
    elif len(response.entities) == 0:
        raise XModalitySExternalError(error=f"app {app_name}/{name} has been registered, but not by {query}")
    return response.entities


class DataLoaderRegisterError(XModalitySExternalError):
    def __init__(self, error: str):
        XModalitySExternalError.__init__(self, error)


class ModelhubRegisterError(XModalitySExternalError):
    def __init__(self, error: str):
        XModalitySExternalError.__init__(self, error)


class DataLoader(abc.ABC):
    @abc.abstractmethod
    def read_images(self, data_source: DataSource, *args, **kwargs) -> ImageFrameDataset:
        raise NotImplementedError

    @abc.abstractmethod
    def read(self, data_source: DataSource, *args, **kwargs) -> Dataset:
        raise NotImplementedError

    # @abc.abstractmethod
    # def read_texts(self, data_source: DataSource, *args, **kwargs) -> ImageFrameDataset:
    #     raise NotImplementedError


@serve.deployment
class DataLoaderSpawnerServer:
    def __init__(self,
                 dataloader: serve.handle.DeploymentHandle | ray.actor.ActorClass,
                 **config):
        self.dataloader = dataloader
        self.config = config

    def spawn(self, *args, **kwargs) -> ray.actor.ActorHandle | serve.handle.DeploymentHandle:
        if isinstance(self.dataloader, serve.handle.DeploymentHandle):
            return self.dataloader
        if self.config is None or len(self.config) == 0:
            return self.dataloader.remote()
        return self.dataloader.remote(**self.config)


class _DataLoaderRegister:
    def __init__(self, name: str, app_name: str, raw: RawStore):
        from loguru import logger
        self.logger = logger
        self.app_name = app_name
        self.name = name
        self.raw = raw
        self.registry = serve.get_deployment_handle(deployment_name=DATAHUB_REGISTRY, app_name=DATAHUB_APP)


    def register_as_actor(self, cls, upgrade:bool=False, ray_actor_options: Dict[str, Any] = None,
                          **user_config) -> serve.handle.DeploymentHandle:
        self._check(cls, upgrade)
        options = ray_actor_options if ray_actor_options is not None else {}

        dataloader = ray.remote(cls, **options)
        return self._create_spawner(dataloader, **user_config)

    def register_as_deployment(self, cls, upgrade:bool=False, ray_deployment_options:Dict[str,Any]=None,
                               **user_config) -> serve.handle.DeploymentHandle:
        self._check(cls, upgrade)
        options = ray_deployment_options if ray_deployment_options is not None else {}
        options["use_new_handle_api"] = True
        dataloader = serve.deployment(cls).options(**options).bind(**user_config)
        return self._create_spawner(dataloader)

    def _create_spawner(self,
                        dataloader: ray.actor.ActorClass | serve.handle.DeploymentHandle,
                        **user_config)->serve.handle.DeploymentHandle:
        app = DataLoaderSpawnerServer.options(name=self.name).bind(dataloader=dataloader, **user_config)
        serve.run(app, name=self.app_name, route_prefix="/" + self.app_name)
        result = self.registry.register.remote(
            RegisterDataLoader(name=self.name, app_name=self.app_name, raw=self.raw)).result()

        if result.error is not None:
            raise XModalitySInternalError(f"regester error {result.error}")
        return serve.get_deployment_handle(deployment_name=self.name, app_name=self.app_name)
    
    def _check(self, cls, upgrade=False):
        if not inspect.isclass(cls):
            raise DataLoaderRegisterError(error="decorated obejct should be class DataLoader")
        
        exists = True
        try:
            exists = self._check_dataloader_exists()
        except XModalitySExternalError as e:
            if upgrade:
                self.logger.warning(f"register with {e}, try forcing upgrade.")
            else:
                raise DataLoaderRegisterError(error=str(e))

        if exists and not upgrade:
            raise DataLoaderRegisterError(
                error=f"dataloader {self.app_name}/{self.name} store {self.raw} has been registered. " +
                      "set upgrade True to overwrite it.")

    def _check_dataloader_exists(self) -> bool:
        try:
            dataloader:RegisterDataLoader = _check_resource_exists(self.app_name, self.name, self.raw, self.registry)
        except XModalitySExternalError as e:
            raise DataLoaderRegisterError(error=str(e))
        
        if dataloader is None :
            return False
        
        if dataloader.app_name != self.app_name or dataloader.name != self.name:
            raise DataLoaderRegisterError(
                error=f"app {self.app_name}/{self.name} has been registered by {dataloader.raw}, not input {self.raw}")

        if dataloader.raw.name != self.raw.name or dataloader.raw.kind != self.raw.kind:
            raise DataLoaderRegisterError(
                error=f"app {self.app_name}/{self.name} has been registered by {dataloader.raw}, not input {self.raw}")
        return True


def dataloader(raw: RawStore,
               name: str | None  = None, app_name: str | None = None, 
               upgrade=False, deployment=False,
               ray_actor_options:Dict[str,Any] = None, ray_deployment_options: Dict[str,Any] = None,
               ) -> serve.handle.DeploymentHandle:
    name = name if name is not None else str(uuid.uuid4())
    app_name = app_name if app_name is not None else str(uuid.uuid4())
    def decorator(cls):
        @functools.wraps(cls)
        def wrapper(**user_config):
            register = _DataLoaderRegister(name=name, app_name=app_name, raw=raw)
            if deployment:
                return register.register_as_deployment(cls, upgrade=upgrade, ray_deployment_options=ray_deployment_options, **user_config)
            return register.register_as_actor(cls, upgrade=upgrade, ray_actor_options=ray_actor_options, **user_config) 
        return wrapper
    return decorator


@serve.deployment
class ModalityModelSpawnerServer:
    def __init__(self,
                 model: serve.handle.DeploymentHandle | ray.actor.ActorClass,
                 **config):
        self.model = model
        self.config = config

    def spawn(self) -> ray.actor.ActorHandle | serve.handle.DeploymentHandle:
        if isinstance(self.model, serve.handle.DeploymentHandle):
            return self.model
        if self.config is None or len(self.config) == 0:
            return self.model.remote()
        return self.model.remote(**self.config)


class _ModalityModelRegister:
    def __init__(self, name: str, app_name: str, model: ModelCard):
        from loguru import logger
        self.logger = logger
        self.app_name = app_name
        self.name = name
        self.model = model
        self.registry = serve.get_deployment_handle(deployment_name=MODELHUB_REGISTRY, app_name=MODELHUB_APP)

    def register_as_actor(self, cls, upgrade:bool=False, ray_actor_options: Dict[str, Any] = None,
                          **user_config) -> serve.handle.DeploymentHandle:
        self._check(cls, upgrade)
        options = ray_actor_options if ray_actor_options is not None else {}

        model = ray.remote(cls, **options)
        return self._create_spawner(model, **user_config)

    def register_as_deployment(self, cls, upgrade:bool=False, ray_deployment_options:Dict[str,Any]=None,
                               **user_config) -> serve.handle.DeploymentHandle:
        self._check(cls, upgrade)
        options = ray_deployment_options if ray_deployment_options is not None else {}
        options["use_new_handle_api"] = True
        model = serve.deployment(cls).options(**options).bind(**user_config)
        return self._create_spawner(model)

    def _create_spawner(self,
                        model: ray.actor.ActorClass | serve.handle.DeploymentHandle,
                        **user_config)->serve.handle.DeploymentHandle:
        app = ModalityModelSpawnerServer.options(name=self.name).bind(model=model, **user_config)
        serve.run(app, name=self.app_name, route_prefix="/" + self.app_name)
        result = self.registry.register.remote(
            RegisterModelSpawner(name=self.name, app_name=self.app_name, model=self.model)).result()

        if result.error is not None:
            raise XModalitySInternalError(f"regester error {result.error}")
        return serve.get_deployment_handle(deployment_name=self.name, app_name=self.app_name)

    def _check(self, cls, upgrade=False):
        if not inspect.isclass(cls):
            raise ModelhubRegisterError(error="decorated obejct should be class ModalityModel")
        
        exists = True
        try:
            exists = self._check_model_exists()
        except XModalitySExternalError as e:
            if upgrade:
                self.logger.warning(f"register with {e}, try forcing upgrade.")
            else:
                raise ModelhubRegisterError(error=str(e))

        if exists and not upgrade:
            raise ModelhubRegisterError(
                error=f"modality model {self.app_name}/{self.name} info {self.model} has been registered. " +
                      "set upgrade True to overwrite it.")

    def _check_model_exists(self) -> bool:
        try:
            model_cards:List[ModelCard] = _check_resource_exists(self.app_name, self.name, self.model, self.registry)
        except XModalitySExternalError as e:
            raise ModelhubRegisterError(error=str(e))
        
        if model_cards is None :
            return False

        if self.model.revision is None:
            self.model.revision = "default"

        revision_map = {}
        for model in model_cards:
            revision = model.revision if model.revision is not None else "default"
            revision_map[revision] = model
        
        value = revision_map.get(self.model.revision, None)
        if value is None:
            return False
        
        if value.app_name != self.app_name or value.name != self.name:
            raise ModelhubRegisterError(
                error=f"modality model {self.app_name}/{self.name} info {self.model} has been registered.")
        return True


def modality_model(model: ModelCard,
                   name: str | None  = None, app_name: str | None = None, 
                   upgrade=False, deployment=False,
                   ray_actor_options:Dict[str,Any] = None, ray_deployment_options: Dict[str,Any] = None,
                   ) -> serve.handle.DeploymentHandle:
    name = name if name is not None else str(uuid.uuid4())
    app_name = app_name if app_name is not None else str(uuid.uuid4())
    def decorator(cls):
        @functools.wraps(cls)
        def wrapper(**user_config):
            register = _ModalityModelRegister(name=name, app_name=app_name, model=model)
            if deployment:
                return register.register_as_deployment(cls, upgrade=upgrade, ray_deployment_options=ray_deployment_options, **user_config)
            return register.register_as_actor(cls, upgrade=upgrade, ray_actor_options=ray_actor_options, **user_config) 
        return wrapper
    return decorator