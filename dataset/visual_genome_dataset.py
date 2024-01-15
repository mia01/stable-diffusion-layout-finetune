from dataset.condition_builders import ObjectsBoundingBoxConditionalBuilder
from dataset.models import Annotation, Category, ImageDescription
from dataset.transforms import CenterCropReturnCoordinates, RandomHorizontalFlipReturn
from utils.helpers import convert_pil_to_tensor, load_image_from_disk, load_json
from collections import defaultdict
from typing import Dict, Iterable, List
from torch.utils.data import Dataset
from torchvision import transforms
# see https://github.com/CompVis/latent-diffusion/issues/260#issuecomment-1526741772
from tqdm import tqdm
from torchvision.transforms import  PILToTensor
from accelerate.logging import get_logger

class VisualGenomeBase(Dataset):
    def __init__(self,
                 split,
                 annotation_json_path,
                 captions_json_path,
                 images_folder,
                 resolution=64
                 ):

        # Load json files
        self.resolution = resolution
        self.annotations_json = load_json(annotation_json_path)
        self.captions_json = load_json(captions_json_path)
        self.logger = get_logger(__name__, log_level="INFO")

        # Convert training data into ImageDescription objects
        self.image_descriptions = self.load_image_descriptions(self.annotations_json["images"])

        self.split = split
        self.images_folder = images_folder

        # Load object category data into Category objects
        self.categories = self.load_categories(self.annotations_json["categories"])

        self.category_id_to_number_dict = self.get_category_id_to_number_dict()

        # Convert training data into Annotation objects
        self.annotations = self.load_annotations(self.annotations_json["annotations"], self.image_descriptions,
                                        self.category_id_to_number_dict, self.split)

        # check if this is needed
        # self.annotations = self.filter_object_number(annotations,  min_object_area =10, min_objects_per_image = 3, max_objects_per_image = 30)

        # Only include image ids that have annotations
        self.image_id_index = sorted(list(self.annotations.keys()))

        self.captions = self.load_captions(self.captions_json["annotations"])

        self.logger.info(f"{len(self.image_id_index)} {split} images loaded")

    def __len__(self):
        return len(self.image_id_index)

    def __getitem__(self, n: int):
        # Get image id by index n
        image_id = self.image_id_index[n]

        # get image annotations
        image_data = self.image_descriptions[image_id]._asdict()
        image_data['annotations'] = self.annotations[image_id]

        # load image
        image_data["original_image"] = load_image_from_disk(f"{self.images_folder}/{image_id}.jpg")
        image_data["tensor_image"] = convert_pil_to_tensor(image_data["original_image"] )
        # sample['image'].permute(1, 2, 0) - do we need this??

        # transform image
        transform_data = self.apply_image_transforms(image_data["original_image"])

        image_data.update(transform_data)

        # get image captions
        image_data["captions"] = self.captions[image_id]

        # apply conditional builders
        bounding_box_data = self.apply_bounding_boxes(self.annotations[image_id], transform_data["crop_bounding_box"], transform_data["flipped"])
        image_data["bounding_boxes"] = bounding_box_data["bounding_boxes"]

        return image_data
    
    
    def get_image_transforms(self, resolution):
        image_transforms = [
                {"tensor_image": PILToTensor()},
                {"resized_image": transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)},
                {"cropped_image": CenterCropReturnCoordinates(resolution)},
                {"flipped_image": RandomHorizontalFlipReturn()},
                {"normalised_image": transforms.Lambda(lambda x: x / 127.5 - 1.)},
            ]

        return image_transforms

    def apply_image_transforms(self, image):

        image_transforms = self.get_image_transforms(self.resolution)
        results = {}
        for transform in image_transforms:
            for key in transform.keys():
                if key == "cropped_image":
                    crop_bbox, image = transform[key](image)
                    results["crop_bounding_box"] = crop_bbox
                elif key == "flipped_image":
                    flipped, image = transform[key](image)
                    results["flipped"] = flipped
                else:
                    image = transform[key](image)

                results[key] = image

        return results
    
    def get_bounding_box_builders(self):
        builders = [{"bounding_boxes": ObjectsBoundingBoxConditionalBuilder(
                    self.categories,
                    no_object_classes = len(self.categories), # the number of object categories
                    no_max_objects = 30,
                    no_tokens = 1024,
                    encode_crop = True,
                    use_group_parameter = True,
                    use_additional_parameters = False
                )}]
        return builders
    
    def apply_bounding_boxes(self, annotations, crop_bounding_box, flipped):
        results = {}
        builders = self.get_bounding_box_builders()
        for builder in builders:
            for key in builder.keys():
                results[key] = builder[key].build(annotations, crop_bounding_box, flipped)

        return results
    
    def load_image_descriptions(self, description_json: List[Dict]):
        image_descriptions_dict =  {}

        for img in description_json:
            image_descriptions_dict[str(img['id'])] =  ImageDescription(
                id=img['id'],
                license=img.get('license'),
                file_name=img['file_name'],
                coco_url=img['coco_url'],
                original_size=(img['width'], img['height']),
                date_captured=img.get('date_captured'),
                flickr_url=img.get('flickr_url')
            )

        return image_descriptions_dict


    def load_categories(self, category_json: Iterable) -> Dict[str, Category]:
        return {str(cat['id']): Category(id=str(cat['id']), super_category=cat['supercategory'], name=cat['name'])
                for cat in category_json if cat['name'] != 'other'}

    
    def get_category_id_to_number_dict(self):
        category_ids = list(self.categories.keys())
        category_ids.sort()
        category_number = {category_id: i for i, category_id in enumerate(category_ids)}
        return category_number
    
    def get_category_no_to_id_dict(self):
        category_ids = list(self.categories.keys())
        category_ids.sort()
        category_number = {i:category_id for i, category_id in enumerate(category_ids)}
        return category_number
    
    def load_annotations(self, annotations_json: List[Dict], image_descriptions, category_id_to_number_dict, split: str) -> Dict[str, List[Annotation]]:
        annotations = defaultdict(list)
        total = len(annotations_json)
        self.logger.info(f"Loading {total} {split} annotations ")
        count = 0
        unprocessed = 0
        for ann in tqdm(annotations_json, f'Loading {split} annotations', total=total):

            image_id = str(ann['image_id'])
            category_id = ann['category_id']
            if image_id not in image_descriptions:
                self.logger.warn(f'image_id [{image_id}] has no image description.')
                unprocessed = unprocessed + 1
                continue
            width, height = image_descriptions[image_id].original_size

            # Normalization of Bounding Box Coordinates
            bbox = (ann['bbox'][0] / width, ann['bbox'][1] / height, ann['bbox'][2] / width, ann['bbox'][3] / height)

            annotations[image_id].append(
                Annotation(
                    id=ann['id'],
                    area=bbox[2]*bbox[3],  # use bbox area
                    is_group_of=ann['iscrowd'],
                    image_id=ann['image_id'],
                    bbox=bbox,
                    category_id=str(category_id),
                    category_no=category_id_to_number_dict[str(category_id)]
                )
            )
            count = count + 1

        self.logger.info(f"{count} {split} annotations processed")
        self.logger.info(f"{unprocessed} {split} annotations unprocessed")
        self.logger.info(f"{len(annotations)} {split} annotations loaded")
        return dict(annotations)


    def load_captions(self, caption_data):
        img_id_to_caption_mapping = dict()
        for ann in caption_data:
            caption = ann['caption'].replace('.', '')
            try:
                img_id_to_caption_mapping[str(ann['image_id'])].append(caption)
            except:
                img_id_to_caption_mapping[str(ann['image_id'])] = [caption]
        return img_id_to_caption_mapping
    

class VisualGenomeTrain(VisualGenomeBase):
    def __init__(self,
                 annotation_json_path,
                 captions_json_path,
                 images_folder,
                 resolution
                 ):
        super().__init__("Train",
                 annotation_json_path,
                 captions_json_path,
                 images_folder,
                 resolution)


class VisualGenomeValidation(VisualGenomeBase):
    def __init__(self,
                 annotation_json_path,
                 captions_json_path,
                 images_folder,
                 resolution
                 ):
        super().__init__("Validation",
                 annotation_json_path,
                 captions_json_path,
                 images_folder,
                 resolution)

class VisualGenomeTest(VisualGenomeBase):
    def __init__(self,
                 annotation_json_path,
                 captions_json_path,
                 images_folder,
                 ):
        super().__init__("Test",
                 annotation_json_path,
                 captions_json_path,
                 images_folder)