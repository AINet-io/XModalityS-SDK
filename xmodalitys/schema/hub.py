# © Copyright 2023, Liu Yuchen, All rights reserved.

import abc
import inspect

import ray
from ray import serve

from xmodalitys.schema.api import DataSource, ImageFrameDataset, Dataset, RegistryGeneralResponse
from xmodalitys.schema.api import RegisterDataLoader
from xmodalitys.schema.entity import RawStore
from xmodalitys.schema.error import XModalitySExternalError, XModalitySInternalError

DATAHUB_REGISTRY = "DatahubRegistry"
DATAHUB_APP = "DatahubRegistry"


class DataLoaderRegisterError(XModalitySExternalError):
    def __init__(self, error: str):
        XModalitySExternalError.__init__(self, error)


class DataLoader(abc.ABC):
    @abc.abstractmethod
    def __init__(self, data_source: DataSource, *args, **kwargs):
        pass

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
                 dataloader: ray.remote_function.RemoteFunction | ray.actor.ActorClass,
                 *args,
                 **kwargs):
        self.dataloader = dataloader

    def spawn(self, data_source: DataSource, *args, **kwargs) -> ray.actor.ActorHandle:
        return self.dataloader.remote(data_source, *args, **kwargs)


class _DataLoaderRegister:
    def __init__(self, name: str, app_name: str, raw: RawStore):
        self.app_name = app_name
        self.name = name
        self.raw = raw
        self.registry = serve.get_deployment_handle(deployment_name=DATAHUB_REGISTRY, app_name=DATAHUB_APP)

    def __call__(self, cls: DataLoader, upgrade=False, *args, **kwargs) -> serve.handle.DeploymentHandle:
        if not inspect.isclass(cls):
            raise DataLoaderRegisterError(error="decorated obejct should be class DataLoader")

        if self._check_dataloader_exists() and not upgrade:
            raise DataLoaderRegisterError(
                error=f"dataloader {self.app_name}/{self.name} store {self.raw} has been registered. " +
                      "set upgrade True to overwrite it.")

        dataloader = ray.remote(cls, *args, **kwargs)
        app = DataLoaderSpawnerServer.options(name=self.name).bind(dataloader=dataloader)
        serve.run(app, name=self.app_name, route_prefix="/" + self.app_name)
        result = self.registry.register.remote(
            RegisterDataLoader(name=self.name, app_name=self.app_name, raw=self.raw)).result()
        if result.error is not None:
            raise XModalitySInternalError(f"regester error {result.error}")
        return serve.get_deployment_handle(deployment_name=self.name, app_name=self.app_name)

    def _check_dataloader_exists(self) -> bool:
        try:
            _ = serve.get_deployment_handle(deployment_name=self.name, app_name=self.app_name)
        except (serve.exceptions.RayServeException, KeyError):
            return False

        response: RegistryGeneralResponse = self.registry.get.remote(self.raw).result()

        if response.error is not None:
            raise XModalitySInternalError(error=f"get dataloader error: {response.error}")

        try:
            dataloader: RegisterDataLoader = response.entities[0]
        except Exception:
            raise DataLoaderRegisterError(
                error=f"app {self.app_name}/{self.name} has been registered, but not by {self.raw}")

        if dataloader.raw.name != self.raw.name or dataloader.raw.kind != self.raw.kind:
            raise DataLoaderRegisterError(
                error=f"app {self.app_name}/{self.name} has been registered by {dataloader.raw}, not input {self.raw}")
        return True


def register_dataloader(cls: DataLoader, name: str, app_name: str, raw: RawStore,
                        *args, upgrade=False, **kwargs) -> serve.handle.DeploymentHandle:
    register = _DataLoaderRegister(name=name, app_name=app_name, raw=raw)
    return register(cls, upgrade=upgrade, *args, **kwargs)
