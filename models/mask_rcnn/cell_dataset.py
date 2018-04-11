import os
import numpy as np
from skimage import io
from skimage.restoration import denoise_bilateral
from skimage import transform

from models.mask_rcnn import utils

class CellsDataset(utils.Dataset):
    def load_cells(self, ids):
        # Add classes
        self.add_class("cells", 1, "cell")

        # Add images
        for i, pathar in enumerate(ids):
            path = pathar[0]
            id_ = pathar[1]
            composed_path = next(os.walk(os.path.join(path,'images')))
            composed_path = os.path.join(composed_path[0], composed_path[2][0])
            self.add_image("cells", image_id=i, path=composed_path, simple_path=path)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        """
        info = self.image_info[image_id]
        img = io.imread(info["path"])
        if len(img.shape) > 2:
            img = img[:,:,:3]
        else:
            img = np.repeat(img[:,:,None], 3, axis=2)

        # upscaling
        height = img.shape[0]
        width = img.shape[1]
        img = transform.resize(img, (height * 2, width * 2))
        img = (img*255).astype('uint8')

        return img

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for cells of the given image ID.
        """
        info = self.image_info[image_id]
        path = info["simple_path"]

        masks = []
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = io.imread(path + '/masks/' + mask_file)
            height = mask_.shape[0]
            width = mask_.shape[1]
            mask_ = transform.resize(mask_, (height * 2, width * 2))
            mask_ = (mask_ > 0).astype(np.uint8)
            masks.append(mask_[...,None])

        count = len(masks)
        masks = np.concatenate(masks, axis=2)
        #masks = masks / 255

        # Map class names to class IDs.
        class_ids = np.array([1 for i in range(count)])
        return masks, class_ids.astype(np.int32)

