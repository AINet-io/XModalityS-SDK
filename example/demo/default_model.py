# © Copyright 2023, Liu Yuchen, All rights reserved.

import skimage

import ray
from ray.serve.handle import DeploymentHandle
from ray import serve

from xmodalitys.schema import RawStore, DataSource, MultimodalRetrieval
from xmodalitys.schema import InsertDatasets
from example.dataloader import RegisteredLocalDataLoader


def retrieve_xmodalitys() -> DeploymentHandle:
    entrypoint = serve.get_deployment_handle(app_name="XModalityS", deployment_name="XModalityS")
    return entrypoint

def test_insert(data_source: DataSource, entry=None):
    entry = retrieve_xmodalitys() if entry is None else entry
    para = InsertDatasets(data_source=data_source)
    return entry.insert_images.remote(para).result()


def test_retrieve(text, entry=None):
    entry = retrieve_xmodalitys() if entry is None else entry
    para = MultimodalRetrieval(
        text=text,
        task="ZeroImageRetrieval",
        start=0,
        count=10,
        projector=["metadata"],
    )
    return entry.multimodal_retrieval.remote(para).result()


descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse",
    "coffee": "a cup of coffee on a saucer"
}
texts = ["This is " + desc for desc in descriptions.values()]

entry = retrieve_xmodalitys()
entry.check_health.remote().result()
raw = RawStore(name="LocalDataLoader", kind="LocalPath", path=skimage.data_dir)
data_source = DataSource(name="skimage_data", raw=raw)
result = test_insert(data_source, entry)
result = test_retrieve(texts[0], entry)
print(f"test insert result {result}")
