import numpy as np
import scipy.stats as stats
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random
import collections
import cv2
import torch
from torchvision import transforms


# # # # # # # # # # # # # # # # # # # # # # # # 
# # # 1. Augmentation for image and labels
# # # # # # # # # # # # # # # # # # # # # # # # 
class Compose(object):
    def __init__(self, segtransforms):
        self.segtransforms = segtransforms

    def __call__(self, image, label):
        for idx, t in enumerate(self.segtransforms):
            if isinstance(t, strong_img_aug):
                image = t(image)
            else:
                image, label = t(image, label)
        return image, label


class ToTensorAndNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        assert len(mean) == len(std)
        assert len(mean) == 3
        self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, in_image, in_label):
        in_image = Image.fromarray(np.uint8(in_image))
        image = self.normalize(self.to_tensor(in_image))
        label = torch.from_numpy(np.array(in_label, dtype=np.int32)).long()

        return image, label


class Resize(object):
    def __init__(self, base_size, ratio_range, scale=True, bigger_side_to_base_size=True):
        # assert isinstance(ratio_range, collections.Iterable) and len(ratio_range) == 2 
        assert isinstance(ratio_range, collections.abc.Iterable) and len(ratio_range) == 2 # for recent python version
        self.base_size = base_size
        self.ratio_range = ratio_range
        self.scale = scale
        self.bigger_side_to_base_size = bigger_side_to_base_size

    def __call__(self, in_image, in_label):
        w, h = in_image.size
        
        if isinstance(self.base_size, int):
            # obtain long_side
            if self.scale:
                long_side = random.randint(int(self.base_size * self.ratio_range[0]), 
                                        int(self.base_size * self.ratio_range[1]))
            else:
                long_side = self.base_size
                
            # obtain new oh, ow
            if self.bigger_side_to_base_size:
                if h > w:
                    oh = long_side
                    ow = int(1.0 * long_side * w / h + 0.5)
                else:
                    oh = int(1.0 * long_side * h / w + 0.5)
                    ow = long_side
            else:
                oh, ow = (long_side, int(1.0 * long_side * w / h + 0.5)) if h < w else (
                        int(1.0 * long_side * h / w + 0.5), long_side)
                
            image = in_image.resize((ow, oh), Image.BILINEAR)
            label = in_label.resize((ow, oh), Image.NEAREST)
            return image, label
        elif (isinstance(self.base_size, list) or isinstance(self.base_size, tuple)) and len(self.base_size) == 2:
            if self.scale:
                # scale = random.random() * 1.5 + 0.5  # Scaling between [0.5, 2]
                scale = self.ratio_range[0] + random.random() * (self.ratio_range[1] - self.ratio_range[0])
                # print("="*100, h, self.base_size[0])
                # print("="*100, w, self.base_size[1])
                oh, ow = int(self.base_size[0] * scale), int(self.base_size[1] * scale)
            else:
                oh, ow = self.base_size
            image = in_image.resize((ow, oh), Image.BILINEAR)
            label = in_label.resize((ow, oh), Image.NEAREST)
            # print("="*100, in_image.size, image.size)
            return image, label

        else:
            raise ValueError


class Crop(object):
    def __init__(self, crop_size, crop_type="rand", mean=[0.485, 0.456, 0.406], ignore_value=255):
        if (isinstance(crop_size, list) or isinstance(crop_size, tuple)) and len(crop_size) == 2:
            self.crop_h, self.crop_w = crop_size
        elif isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        else:
            raise ValueError
        
        self.crop_type = crop_type
        self.image_padding = (np.array(mean) * 255.).tolist()
        self.ignore_value = ignore_value

    def __call__(self, in_image, in_label):
        # Padding to return the correct crop size
        w, h = in_image.size
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT, 
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(np.asarray(in_image, dtype=np.float32), 
                                       value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(np.asarray(in_label, dtype=np.int32), 
                                       value=self.ignore_value, **pad_kwargs)
            image = Image.fromarray(np.uint8(image))
            label = Image.fromarray(np.uint8(label))
        else:
            image = in_image
            label = in_label
        
        # cropping
        w, h = image.size
        if self.crop_type == "rand":
            x = random.randint(0, w - self.crop_w)
            y = random.randint(0, h - self.crop_h)
        else:
            x = (w - self.crop_w) // 2
            y = (h - self.crop_h) // 2
        image = image.crop((x, y, x + self.crop_w, y + self.crop_h))
        label = label.crop((x, y, x + self.crop_w, y + self.crop_h))
        return image, label


class RandomFlip(object):
    def __init__(self, prob=0.5, flag_hflip=True,):
        self.prob = prob
        if flag_hflip:
            self.type_flip = Image.FLIP_LEFT_RIGHT
        else:
            self.type_flip = Image.FLIP_TOP_BOTTOM
            
    def __call__(self, in_image, in_label):
        if random.random() < self.prob:
            in_image = in_image.transpose(self.type_flip)
            in_label = in_label.transpose(self.type_flip)
        return in_image, in_label


# # # # # # # # # # # # # # # # # # # # # # # #
# # # 2. Strong Augmentation for image only
# # # # # # # # # # # # # # # # # # # # # # # #

import math

def _ret(img_out, t, return_param: bool):
    if return_param:
        return img_out, float(t) if t is not None else float("nan")
    return img_out


def img_aug_identity(img, scale=None, return_param=False):
    return _ret(img, float("nan"), return_param)


def img_aug_autocontrast(img, scale=None, return_param=False):
    out = ImageOps.autocontrast(img)
    return _ret(out, float("nan"), return_param)


def img_aug_equalize(img, scale=None, return_param=False):
    out = ImageOps.equalize(img)
    return _ret(out, float("nan"), return_param)


def img_aug_invert(img, scale=None, return_param=False):
    out = ImageOps.invert(img)
    return _ret(out, float("nan"), return_param)


def img_aug_blur(img, scale=[0.1, 2.0], return_param=False):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])  # t = sigma
    out = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return _ret(out, sigma, return_param)


def img_aug_contrast(img, scale=[0.05, 0.95], return_param=False):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v  # t = v
    out = ImageEnhance.Contrast(img).enhance(v)
    return _ret(out, v, return_param)


def img_aug_brightness(img, scale=[0.05, 0.95], return_param=False):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v  # t = v
    out = ImageEnhance.Brightness(img).enhance(v)
    return _ret(out, v, return_param)


def img_aug_color(img, scale=[0.05, 0.95], return_param=False):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v  # t = v
    out = ImageEnhance.Color(img).enhance(v)
    return _ret(out, v, return_param)


def img_aug_sharpness(img, scale=[0.05, 0.95], return_param=False):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v  # t = v
    out = ImageEnhance.Sharpness(img).enhance(v)
    return _ret(out, v, return_param)


def img_aug_hue(img, scale=[0, 0.5], return_param=False):
    min_v, max_v = min(scale), max(scale)
    v_mag = float(max_v - min_v) * random.random()
    v_mag += min_v
    hue_factor = -v_mag if (np.random.random() < 0.5) else v_mag  # t = hue_factor

    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        # ảnh grayscale thì hue không áp dụng được
        return _ret(img, hue_factor, return_param)

    h, s, v_chan = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)

    # uint8 addition takes care of rotation across boundaries
    with np.errstate(over="ignore"):
        delta = int(round(hue_factor * 255))
        np_h = (np_h.astype(np.int16) + delta) % 256
        np_h = np_h.astype(np.uint8)

    h2 = Image.fromarray(np_h, "L")
    out = Image.merge("HSV", (h2, s, v_chan)).convert(input_mode)
    return _ret(out, hue_factor, return_param)


def img_aug_posterize(img, scale=[4, 8], return_param=False):
    # PIL posterize dùng "bits" (số bit giữ lại). bits càng nhỏ => càng mạnh
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    bits = max_v - v  # t = bits (discrete)
    out = ImageOps.posterize(img, bits)
    return _ret(out, bits, return_param)


def img_aug_solarize(img, scale=[1, 256], return_param=False):
    # threshold càng thấp => càng mạnh
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = int(np.ceil(v))
    v = max(1, v)
    thr = max_v - v  # t = thr (discrete threshold)
    out = ImageOps.solarize(img, thr)
    return _ret(out, thr, return_param)


def get_augment_list(flag_using_wide=False):
    if flag_using_wide:
        l = [
            (1, img_aug_identity, None),
            (2, img_aug_autocontrast, None),
            (3, img_aug_equalize, None),
            (4, img_aug_blur, [0.1, 2.0]),
            (5, img_aug_contrast, [0.1, 1.8]),
            (6, img_aug_brightness, [0.1, 1.8]),
            (7, img_aug_color, [0.1, 1.8]),
            (8, img_aug_sharpness, [0.1, 1.8]),
            (9, img_aug_posterize, [2, 8]),
            (10, img_aug_solarize, [1, 256]),
            (11, img_aug_hue, [0, 0.5]),
        ]
    else:
        l = [
            (1, img_aug_identity, None),
            (2, img_aug_autocontrast, None),
            (3, img_aug_equalize, None),
            (4, img_aug_blur, [0.1, 2.0]),
            (5, img_aug_contrast, [0.05, 0.95]),
            (6, img_aug_brightness, [0.05, 0.95]),
            (7, img_aug_color, [0.05, 0.95]),
            (8, img_aug_sharpness, [0.05, 0.95]),
            (9, img_aug_posterize, [4, 8]),
            (10, img_aug_solarize, [1, 256]),
            (11, img_aug_hue, [0, 0.5]),
        ]
    return l


class strong_img_aug:
    def __init__(self, num_augs, flag_using_random_num=False):
        assert 1 <= num_augs <= 11
        self.n = num_augs  # số op tối đa (cũng là độ dài pad output)
        self.augment_list = get_augment_list(flag_using_wide=False)
        self.flag_using_random_num = flag_using_random_num

    def __call__(self, img, return_info: bool = False):
        # chọn số op thực sự sẽ áp dụng
        if self.flag_using_random_num:
            max_num = np.random.randint(1, high=self.n + 1)  # 1..n
        else:
            max_num = self.n

        # chọn ops: mỗi phần tử là (k_id, op_fn, scales)
        ops = random.choices(self.augment_list, k=max_num)

        k_ids = []
        t_vals = []

        for k_id, op_fn, scales in ops:
            if return_info:
                img, t = op_fn(img, scales, return_param=True)
                k_ids.append(int(k_id))
                t_vals.append(float(t))
            else:
                img = op_fn(img, scales, return_param=False)

        if return_info:
            # pad cho đủ length = self.n để DataLoader stack được thành [B, n]
            pad_len = self.n - len(k_ids)
            if pad_len > 0:
                k_ids.extend([0] * pad_len)                 # 0 = padding
                t_vals.extend([float("nan")] * pad_len)     # NaN = padding

            return img, k_ids, t_vals

        return img