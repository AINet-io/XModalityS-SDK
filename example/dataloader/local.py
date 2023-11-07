# © Copyright 2023, Liu Yuchen, All rights reserved.

import os
import re
from typing import Dict, Any
import skimage
from ray import serve
import time

import numpy as np
import pandas as pd
from PIL import Image

from xmodalitys.schema import DataLoader, dataloader
from xmodalitys.schema import DataSource, ImageFrameDataset, Dataset, RawStore


class LocalDataLoader(DataLoader):
    def __init__(self):
        pass 
    
    def read_images(self, data_source: DataSource, *args,
                    filename_selector: str | None = None, labels: Dict[str, Any] | None = None,
                    **kwargs) -> ImageFrameDataset:
        if data_source.raw.path is None:
            raise Exception("data source root path should not be None.")

        root = data_source.raw.path
        filenames = os.listdir(root)
        if filename_selector is not None:
            filenames = [filename for filename in filenames if re.match(filename_selector, filename) is not None]

        if data_source.subslice is not None and data_source.subslice.series is not None:
            valid_files = set(data_source.subslice.series)
            filenames = [filename for filename in filenames if filename in valid_files]

        filenames = [filename for filename in filenames if filename.endswith(".png") or filename.endswith(".jpg")]

        images = []
        names = []
        for filename in filenames:
            name = os.path.splitext(filename)[0]

            image = Image.open(os.path.join(root, filename)).convert("RGB")

            images.append(np.array(image))
            names.append(root + "/" + name)
        data = pd.DataFrame({"name": names, "frame": images})
        dataset_name = data_source.name if data_source.name is not None else root
        return ImageFrameDataset(name=dataset_name, frames=data, raw=data_source.raw, labels=labels)

    def read(self, data_source: DataSource, *args, filename_selector: str | None = None, **kwargs) -> Dataset:
        pass


@dataloader(raw=RawStore(name="LocalDataLoader", kind="LocalPath"),
            name="LocalDataLoader", app_name="LocalDataLoader", upgrade=True)
class RegisteredLocalDataLoader(LocalDataLoader):
    def __init__(self):
        pass 
    

def remote_loading_images(data_source):
    RegisteredLocalDataLoader()
    hub = serve.get_deployment_handle(deployment_name="Datahub", app_name="XModalityS")
    results = hub.read_images.remote(data_source).result()
    now = time.time()
    ref = hub.read_images.remote(data_source)
    print(f"type of response {type(ref)}")
    results = ref.result()
    print(f"type of results {type(results)}")
    print(f"loading cost {time.time() - now}")
    return results.dataset

if __name__ == "__main__":
    raw = RawStore(name="LocalDataLoader", kind="LocalPath", path=skimage.data_dir)
    data_source = DataSource(name="skimage_data", raw=raw)
    dataset = remote_loading_images(data_source)
    now = time.time()
    names = dataset.get_names()
    images = dataset.get_images()
    print(f"get_image cost {time.time() - now}")
    print(len(names) == len(images))
    images = dataset.get_images()