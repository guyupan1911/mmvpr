from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd
from pathlib import Path
import numpy as np
import torch

from mmengine.dataset import BaseDataset
from mmvpr.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class GSVCities(BaseDataset):
    def __init__(self,
                 dataset_path=None,
                 cities="all", # or None
                 img_per_place=4,
                 random_sample_from_each_place=True,
                 hard_mining=False,
                 pipeline: Sequence = (),
                 serialize_data: bool = True,
                 lazy_init: bool = False
                 ):
        """
        Args:
            cities (list): List of city names to use in the dataset. Default is "all" or None which uses all cities.
            base_path (Path): Base path for the dataset files.
            img_per_place (int): The number of images per place.
            random_sample_from_each_place (bool): Whether to sample images randomly from each place.
            hard_mining (bool): Whether you are performing hard negative mining or not.
        """
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist. Please check the path.")
        
        self.base_path = Path(dataset_path)
        
        # let's check if the cities are valid
        if cities == "all" or cities is None:
            # get all cities from the Dataframes folder
            cities = [f.name[:-4] for f in self.base_path.glob("Dataframes/*.csv")]
        else:
            for city in cities:
                if not (self.base_path / 'Dataframes' / f'{city}.csv').exists():
                    raise FileNotFoundError(f"Dataframe for city {city} not found. Please check the city name.")

        self.cities = cities
        self.img_per_place = img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.hard_mining = hard_mining

        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transforms.append(TRANSFORMS.build(transform))
            else:
                transforms.append(transform)

        super().__init__(
            pipeline=transforms,
            serialize_data=serialize_data,
        )

        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing one DataFrame
            for each city in self.cities
        '''
        dataframes = []
        for i, city in enumerate(self.cities):
            df = pd.read_csv(self.base_path / 'Dataframes' / f'{city}.csv')
            df['place_id'] += i * 10**5 # to avoid place_id conflicts between cities
            df = df.sample(frac=1) # we always shuffle in city level
            dataframes.append(df)
        
        df = pd.concat(dataframes)
        # keep only places depicted by at least img_per_place images
        df = df[df.groupby('place_id')['place_id'].transform('size') >= self.img_per_place]

        # generate the dataframe contraining images metadata
        self.dataframe = df.set_index('place_id')
        
        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)


    def load_data_list(self):
        self.__getdataframes()
        
        """Load image paths and place_id."""
        data_list = []
        for place_id in self.places_ids:
            place = self.dataframe.loc[place_id]

            if self.random_sample_from_each_place:
                place = place.sample(n=self.img_per_place)
            else:
                place = place.sort_values(
                    by=['year', 'month', 'lat'], ascending=False)
                place = place[: self.img_per_place]
        
            img_paths = []
            for i, row in place.iterrows():
                img_name = self.get_img_name(row)
                img_path = self.base_path / 'Images' / row['city_id'] / img_name
                img_paths.append(img_path)
            info = {'img_paths': img_paths, 'place_id': torch.tensor(place_id).repeat(self.img_per_place)}
            
            data_list.append(info)
        
        return data_list

    @staticmethod
    def get_img_name(row):
        """
            Given a row from the dataframe
            return the corresponding image name
        """
        city = row['city_id']
        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10**5  #row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = f"{city}_{pl_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
        return name


REQUIRED_FILES = {
    "pitts30k-val":     ["pitts30k_val_dbImages.npy", "pitts30k_val_qImages.npy", "pitts30k_val_gt_25m.npy"],
    "pitts30k-test":    ["pitts30k_test_dbImages.npy", "pitts30k_test_qImages.npy", "pitts30k_test_gt_25m.npy"],
    "pitts250k-test":   ["pitts250k_test_dbImages.npy", "pitts250k_test_qImages.npy", "pitts250k_test_gt_25m.npy"],
}

@DATASETS.register_module()
class PittsburghDataset(BaseDataset):
    """
    Pittsburg dataset. It can load pitts30k-val, pitts30k-test and pitts250k-test.

    Args:
        dataset_path (str): Directory containing the dataset. If None, the path in config/data/config.yaml will be used.
        input_transform (callable, optional): Optional transform to be applied on each image.
    """

    def __init__(
        self,
        dataset_path: Optional [str] = None,
        pipeline: Sequence = (),
        lazy_init: bool = False
    ):
        dataset_path = Path(dataset_path)
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"The directory {dataset_path} does not exist. Please check the path.")
        
        if "pitts30k-val" in dataset_path.name:
            self.dataset_name = "pitts30k-val"
        elif "pitts30k-test" in dataset_path.name:
            self.dataset_name = "pitts30k-test"
        elif "pitts250k-test" in dataset_path.name:
            self.dataset_name = "pitts250k-test"
        else:
            raise FileNotFoundError(f"Please make sure the dataset name is either `pitts30k-val`, `pitts30k-test` or `pitts250k-test`.")
        
        # make sure required metadata files are in the directory        
        if not all((dataset_path / file).is_file() for file in REQUIRED_FILES[self.dataset_name]):
            raise FileNotFoundError(f"Please make sure all requiered metadata for {dataset_path} are in the directory. i.e. {REQUIRED_FILES[self.dataset_name]}")
        
        self.dataset_path = dataset_path
        self.dbImages = np.load(dataset_path / REQUIRED_FILES[self.dataset_name][0])
        self.qImages = np.load(dataset_path / REQUIRED_FILES[self.dataset_name][1])
        self.ground_truth = np.load(dataset_path / REQUIRED_FILES[self.dataset_name][2], allow_pickle=True)
  

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        # combine reference and query images
        self.image_paths = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transforms.append(TRANSFORMS.build(transform))
            else:
                transforms.append(transform)

        super().__init__(
            pipeline=transforms,
        )

        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

    def load_data_list(self):
        """Load image paths and place_id."""
        data_list = []
        for index, img_path in enumerate(self.image_paths):
            info = {'img_path': (self.dataset_path / img_path),
                    'num_references': self.num_references,
                    'num_queries': self.num_queries}
            if index - self.num_references >= 0:
                info.update({'gt': self.ground_truth[index - self.num_references]})        
            data_list.append(info)

        return data_list


@DATASETS.register_module()
class MapillarySLSDataset(BaseDataset):
    """
    MapillarySLS validation dataset for visual place recognition.

    Args:
        dataset_path (str): Directory containing the dataset. If None, the path in config/data/config.yaml will be used.
        input_transform (callable, optional): Optional transform to be applied on each image.
        
    Reference:
        @inProceedings{Warburg_CVPR_2020,
        author    = {Warburg, Frederik and Hauberg, Soren and Lopez-Antequera, Manuel and Gargallo, Pau and Kuang, Yubin and Civera, Javier},
        title     = {Mapillary Street-Level Sequences: A Dataset for Lifelong Place Recognition},
        booktitle = {Computer Vision and Pattern Recognition (CVPR)},
        year      = {2020},
        month     = {June}
        }
    """

    def __init__(
        self,
        dataset_path: Optional [str] = None,
        pipeline: Sequence = (),
        lazy_init: bool = False
    ): 
        dataset_path = Path(dataset_path)
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"The directory {dataset_path} does not exist. Please check the path.")

        # make sure the path contains folders `cph` and `sf` and  the 
        # files `msls_val_dbImages.npy`, `msls_val_qImages.npy`, 
        # and `msls_val_gt_25m.npy`
        if not (dataset_path / "cph").is_dir() or not (dataset_path / "sf").is_dir():
            raise FileNotFoundError(f"The directory {dataset_path} does not contain the folders `cph` and `sf`. Please check the path.")
        if not (dataset_path / "msls_val_dbImages.npy").is_file():
            raise FileNotFoundError(f"The file 'msls_val_dbImages.npy' does not exist in {dataset_path}. Please check the path.")
        if not (dataset_path / "msls_val_qImages.npy").is_file():
            raise FileNotFoundError(f"The file 'msls_val_qImages.npy' does not exist in {dataset_path}. Please check the path.")
        if not (dataset_path / "msls_val_gt_25m.npy").is_file():
            raise FileNotFoundError(f"The file 'msls_val_gt_25m.npy' does not exist in {dataset_path}. Please check the path.")
        
        self.dataset_name = "msls-val"
        self.dataset_path = dataset_path
        # Load image names and ground truth data
        self.dbImages = np.load(dataset_path / "msls_val_dbImages.npy")
        self.qImages = np.load(dataset_path / "msls_val_qImages.npy")
        self.ground_truth = np.load(dataset_path / "msls_val_gt_25m.npy", allow_pickle=True)

        # Combine reference and query images
        self.image_paths = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transforms.append(TRANSFORMS.build(transform))
            else:
                transforms.append(transform)

        super().__init__(
            pipeline=transforms,
        )

        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()


    def load_data_list(self):
        """Load image paths and place_id."""
        data_list = []
        for index, img_path in enumerate(self.image_paths):
            info = {'img_path': (self.dataset_path / img_path),
                    'num_references': self.num_references,
                    'num_queries': self.num_queries}
            if index - self.num_references >= 0:
                info.update({'gt': self.ground_truth[index - self.num_references]})        
            data_list.append(info)

        return data_list
