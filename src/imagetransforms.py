import torch
import math
import random
import numpy as np
import numbers
import types
import collections
import cv2
from scipy.interpolate import griddata


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class IdentityTransform(object):
    def __call__(self, img):
        return img

class PickOne(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        idx = np.random.choice(len(self.transforms))
        #print("Calling augmentation of type: %s" % str(type(self.transforms[idx])))
        return self.transforms[idx](img)

class PickRandomParameters(object):
    def __init__(self, cls, params):
        transforms = []
        for p in params:
            transforms.append(cls(**p))

        self.pick = PickOne(transforms)

    def __call__(self, img):
        return self.pick(img)

class Randomize(object):
    def __init__(self, prob, transform):
        self.transform = transform
        self.prob = prob

    def __call__(self, img):
        if torch.bernoulli(torch.Tensor([self.prob]))[0] == 1:
            #print("Calling augmentation of type: %s" % str(type(self.transform)))
            return self.transform(img)
        else:
            return img


class RotateRandom(object):
    def __init__(self, from_angle, to_angle):
        self.from_angle = from_angle
        self.to_angle = to_angle

        self.rand_sampler = torch.distributions.uniform.Uniform(self.from_angle, self.to_angle)

    def __call__(self, img):
        angle = self.rand_sampler.sample().item()

        rows,cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle=angle,scale=1)
        return cv2.warpAffine(img,M,(cols,rows), borderMode=cv2.BORDER_REPLICATE)

class TessBlockConv(object):
    def __init__(self, kernel_val=1, bias_val=1, rand=False):

        self.rand = rand

        with torch.no_grad():
            self.conv_3ch = torch.nn.Conv2d(3, 3, kernel_size=2, padding=1, groups=3)
            self.conv_1ch = torch.nn.Conv2d(1, 1, kernel_size=2, padding=1)

            if rand:
                torch.nn.init.normal_(self.conv_3ch.weight, 0, 1)
                torch.nn.init.normal_(self.conv_3ch.bias, 0, 1)
                torch.nn.init.normal_(self.conv_1ch.weight, 0, 1)
                torch.nn.init.normal_(self.conv_1ch.bias, 0, 1)
            else:
                self.conv_3ch.weight.fill_(kernel_val)
                self.conv_3ch.bias.fill_(bias_val)
                self.conv_1ch.weight.fill_(kernel_val)
                self.conv_1ch.bias.fill_(bias_val)

    def __call__(self, img):
        # If random, want to reset weights each time
        if self.rand:
            with torch.no_grad():
                torch.nn.init.normal_(self.conv_3ch.weight, 0, 1)
                torch.nn.init.normal_(self.conv_3ch.bias, 0, 1)
                torch.nn.init.normal_(self.conv_1ch.weight, 0, 1)
                torch.nn.init.normal_(self.conv_1ch.bias, 0, 1)

        if len(img.shape) == 2:
            # Convert numpy image into torch Tensor to pass through conv layer
            img_tensor = torch.from_numpy(img).float()
            # add batch dimension & channel dimension
            img_tensor.unsqueeze_(0)
            img_tensor.unsqueeze_(0)

            # Now pass through conv layer
            img_tensor = self.conv_1ch(img_tensor)

            # Convert back to numpy and rescale
            img_out = img_tensor.squeeze().detach().numpy()
            img_max = np.max(img_out)
            img_min = np.min(img_out)
            img_out = 255 * (img_out - img_min) / (img_max - img_min)
            img_out = img_out.astype(np.uint8)

        else:
            # Convert numpy image into torch Tensor to pass through conv layer
            img_tensor = torch.from_numpy(img.transpose((2,0,1))).float()
            # add batch dimension
            img_tensor.unsqueeze_(0)

            # Now pass through conv layer
            img_tensor = self.conv_3ch(img_tensor)

            # Convert back to numpy and rescale
            img_out = img_tensor.squeeze().detach().numpy()
            img_out = img_out.transpose(1,2,0)
            img_max = np.max(img_out)
            img_min = np.min(img_out)
            img_out = 255 * (img_out - img_min) / (img_max - img_min)
            img_out = img_out.astype(np.uint8)

        # Now we can finally return
        return img_out
        

class MorphErode(object):
    def __init__(self, k):
        self.kernel_size = k

    def __call__(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(self.kernel_size,self.kernel_size))
        return cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)

class MorphDilate(object):
    def __init__(self, k):
        self.kernel_size = k

    def __call__(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(self.kernel_size,self.kernel_size))
        return cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)

class MorphEllipse(object):
    def __init__(self, k):
        self.kernel_size = k

    def __call__(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(self.kernel_size,self.kernel_size))
        return cv2.morphologyEx(img, cv2.MORPH_ELLIPSE, kernel)


class UniformNoise(object):
    def __init__(self, p_apply, noise_std):
        self.p = p_apply
        self.noise_std_ = noise_std

    def __call__(self, img):
        mask = torch.Tensor(img.shape[0],img.shape[1]).fill_(self.p)
        mask = torch.bernoulli(mask).numpy()
        cpy = img.copy().astype(np.int16)

        noise_mean = 0
        if len(img.shape) == 3:
            noise_std = torch.Tensor(img.shape[0],img.shape[1], img.shape[2]).fill_(self.noise_std_)
        else:
            noise_std = torch.Tensor(img.shape[0],img.shape[1]).fill_(self.noise_std_)
        noise = torch.normal(noise_mean, noise_std).numpy().astype(np.int8)

        cpy[ mask == 1 ] += noise[ mask == 1 ]
        cpy = np.clip(cpy, a_min=0, a_max=255)
        cpy = cpy.astype(np.uint8)

        return cpy


class DegradeDownsample(object):
    def __init__(self, ds_factor):
        self.ds_factor = ds_factor

    def __call__(self, img):
        orig_size = (img.shape[1], img.shape[0])
        downsampled_img = cv2.resize(img, dsize=(0,0), fx=self.ds_factor, fy=self.ds_factor)
        return cv2.resize(downsampled_img, dsize=orig_size)


class UniformDelete(object):
    def __init__(self, p_apply):
        self.p = p_apply

    def __call__(self, img):
        mask = torch.Tensor(img.shape[0],img.shape[1]).fill_(self.p)
        mask = torch.bernoulli(mask)
        cpy = img.copy()
        cpy[ mask.numpy() == 1 ] = 255
        return cpy



class AddErrorPartial(object):
    def __init__(self, width_pct):
        self.width_pct = width_pct

        self.tfm = PickOne([
            MorphDilate(5),
            MorphErode(5),
            WarpImage(w_mesh_interval=40, h_mesh_interval=40, w_mesh_std=5, h_mesh_std=3),
            UniformNoise(p_apply=0.7, noise_std=20),
            UniformDelete(p_apply=0.7),
        ])

    def __call__(self, img):
        # Pick random spot for center
        img_width = img.shape[1]
        cntr = np.random.choice(img_width)
        l = max(0, int(cntr - img_width * self.width_pct/100.))
        r = min(img.shape[1], int(cntr + img_width * self.width_pct/100. + 1))

        # Apply extreme noise to img in region between left and right boundaries
        # 1) erode w/ big dilation
        cpy = img.copy()
        cpy[:,l:r] = self.tfm(img[:,l:r])
        return cpy



_WARP_INTERPOLATION = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC
}
class WarpImage(object):
    def __init__(self, **kwargs):
        self.w_mesh_interval = kwargs.get('w_mesh_interval', 25)
        self.w_mesh_std = kwargs.get('w_mesh_std', 3.0)

        self.h_mesh_interval = kwargs.get('h_mesh_interval', 25)
        self.h_mesh_std = kwargs.get('h_mesh_std', 3.0)

        self.interpolation_method = kwargs.get('interpolation', 'linear')
        self.fit_interval_to_image = kwargs.get("fit_interval_to_image", True)

        self.random_state = np.random.RandomState()


    def __call__(self, img):

        h, w = img.shape[:2]

        w_mesh_interval_ = self.w_mesh_interval
        h_mesh_interval_ = self.h_mesh_interval

        if self.fit_interval_to_image:
            # Change interval so it fits the image size
            w_ratio = w / float(w_mesh_interval_)
            h_ratio = h / float(h_mesh_interval_)

            w_ratio = max(1, round(w_ratio))
            h_ratio = max(1, round(h_ratio))

            w_mesh_interval_ = w / w_ratio
            h_mesh_interval_ = h / h_ratio
            ############################################

        # Get control points
        source = np.mgrid[0:h+h_mesh_interval_:h_mesh_interval_, 0:w+w_mesh_interval_:w_mesh_interval_]
        source = source.transpose(1,2,0).reshape(-1,2)

        # Perturb source control points
        destination = source.copy()
        source_shape = source.shape[:1]
        destination[:,0] = destination[:,0] + self.random_state.normal(0.0, self.h_mesh_std, size=source_shape)
        destination[:,1] = destination[:,1] + self.random_state.normal(0.0, self.w_mesh_std, size=source_shape)

        # Warp image
        grid_x, grid_y = np.mgrid[0:h, 0:w]
        grid_z = griddata(destination, source, (grid_x, grid_y), method=self.interpolation_method).astype(np.float32)
        map_x = grid_z[:,:,1]
        map_y = grid_z[:,:,0]
        warped = cv2.remap(img, map_x, map_y, _WARP_INTERPOLATION[self.interpolation_method], borderValue=(255,255,255))

        return warped


class WarpImageLocal(object):
    def __init__(self, **kwargs):
        self.w_mesh_interval = kwargs.get('w_mesh_interval', 25)
        self.w_mesh_std = kwargs.get('w_mesh_std', 3.0)

        self.h_mesh_interval = kwargs.get('h_mesh_interval', 25)
        self.h_mesh_std = kwargs.get('h_mesh_std', 3.0)

        self.interpolation_method = kwargs.get('interpolation', 'linear')
        self.fit_interval_to_image = kwargs.get("fit_interval_to_image", True)

        self.local_interval = kwargs.get('local_interval', 70)

        self.random_state = np.random.RandomState()


    def __call__(self, img):
        h, w = img.shape[:2]

        warped = np.copy(img)
        
        last_destination = None
        for local_start in range(0, w, self.local_interval):
            w_mesh_interval_local = self.w_mesh_interval
            h_mesh_interval_local = self.h_mesh_interval
            local_end = min(w+1, local_start + self.local_interval)
            local_w = local_end - local_start
            
            if self.fit_interval_to_image:
                # Change interval so it fits the image size
                w_ratio = local_w / float(self.w_mesh_interval)
                h_ratio = h / float(self.h_mesh_interval)

                w_ratio = max(1, round(w_ratio))
                h_ratio = max(1, round(h_ratio))

                w_mesh_interval_local = local_w / w_ratio
                h_mesh_interval_local = h / h_ratio
                ############################################

            # Get control points
            source = np.mgrid[0:h+h_mesh_interval_local:h_mesh_interval_local, local_start:local_end+w_mesh_interval_local:w_mesh_interval_local]

            entries_per_row = source.shape[2]
            num_rows = source.shape[1]
            source = source.transpose(1,2,0).reshape(-1,2)

            # Perturb source control points
            destination = source.copy()
            source_shape = source.shape[:1]
            destination[:,0] = destination[:,0] + self.random_state.normal(0.0, self.h_mesh_std, size=source_shape)
            destination[:,1] = destination[:,1] + self.random_state.normal(0.0, self.w_mesh_std, size=source_shape)
            
            # Make sure seams are continious by making destination[t][0] = destination[t-1][WIDTH]
            if not last_destination is None:
                # First Y coord should match last Y Coord
                for r in range(num_rows):
                    destination[r*entries_per_row, 0] = last_destination[r*last_entries_per_row + (last_entries_per_row - 1), 0]
                    destination[r*entries_per_row, 1] = last_destination[r*last_entries_per_row + (last_entries_per_row - 1), 1]
                
            last_destination = destination.copy()
            last_entries_per_row = entries_per_row
            
            # Warp image
            grid_x, grid_y = np.mgrid[0:h, local_start:local_end]
            grid_z = griddata(destination, source, (grid_x, grid_y), method=self.interpolation_method).astype(np.float32)
            map_x = grid_z[:,:,1]
            map_y = grid_z[:,:,0]
            warped_local = cv2.remap(img, map_x, map_y, _WARP_INTERPOLATION[self.interpolation_method], borderValue=(255,255,255))
            
            warped[:,local_start:local_end] = warped_local[:, :warped[:,local_start:local_end].shape[1]]

        return warped


class InvertBlackWhite(object):
    def __call__(self, img):
        return -img + 255


class AddRandomStripe(object):
    def __init__(self, val, strip_width_from, strip_width_to):
        self.val = val
        self.strip_width_from = strip_width_from
        self.strip_width_to = strip_width_to
        self.rand_sampler = torch.distributions.uniform.Uniform(0, 1)

    def __call__(self, img):
        # Pick a random starting position
        rand_pct = self.rand_sampler.sample().item()
        starting_pt = int(rand_pct * img.shape[0])

        # Pick a random strip width
        strip_width = random.randint(self.strip_width_from, self.strip_width_to)

        ending_pt = min(img.shape[0], starting_pt + strip_width)

        img[starting_pt:ending_pt] = self.val
        return img

class ConvertGray(object):
    def __call__(self, img):
        #print('...shape', img.shape, img.shape[2])
        if len(img.shape) == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            return img

class ConvertColor(object):
    def __call__(self, img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            return img
        else:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        if len(pic.shape) == 2:
            h, w = pic.shape
            return torch.from_numpy(pic).view(1, h, w).float().div(255)
        else:
            h, w, c = pic.shape
            return torch.from_numpy(pic.transpose((2,0,1))).float().div(255)

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


class Scale(object):
    """Rescales the input image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size=None, new_h=None, new_w=None, preserve_apsect_ratio=True, interpolation=cv2.INTER_CUBIC):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2) or isinstance(new_h, int) or isinstance(new_w, int)
        self.size = size
        self.new_h = new_h
        self.new_w = new_w
        self.preserve_apsect_ratio = preserve_apsect_ratio
        self.interpolation = interpolation

    def __call__(self, img):
        if len(img.shape) == 2:
            h, w = img.shape
        else:
            h, w, c = img.shape

        # First check if we specified specific height and/or width
        if not (self.new_h is None and self.new_w is None):
            local_new_h = self.new_h or h
            local_new_w = self.new_w or w

            if self.preserve_apsect_ratio and self.new_h is None:
                local_new_h = int(h * float(self.new_w/w))
            if self.preserve_apsect_ratio and self.new_w is None:
                local_new_w = int(w * float(self.new_h/h))

            if local_new_w <= 0 or local_new_h <= 0:
                print("Warning!! local_new_h = %d, local_new_w = %d; orig h = %d, orig w = %d" % (local_new_h, local_new_w, h, w))
                # let's fallback to have a non-zero width; this will give junk results but at least not crash
                local_new_w = 1

            return cv2.resize(img, (local_new_w, local_new_h), self.interpolation)

        # Next, fall back to old tuple API
        if isinstance(self.size, int):
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return cv2.resize(img, (ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return cv2.resize(img, (ow, oh), self.interpolation)
        else:
            return cv2.resize(img, self.size, self.interpolation)


class CenterCrop(object):
    """Crops the given opencv image at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


class Pad(object):
    """Pads the given opencv image on all sides with the given "pad" value"""

    def __init__(self, h_pad, v_pad, fill=0):
        assert isinstance(h_pad, numbers.Number)
        assert isinstance(v_pad, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.h_pad = h_pad
        self.v_pad = v_pad
        self.fill = fill

    def __call__(self, img):
        return cv2.copyMakeBorder(img, top=self.v_pad, bottom=self.v_pad, left=self.h_pad, right=self.h_pad, borderType=cv2.BORDER_CONSTANT, value=self.fill)


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class Identity(object):
    def __call__(self, img):
        return img

class PadRandom(object):
    def __init__(self, pxl_max_horizontal, pxl_max_vertical):
        self.pxl_max_horizontal = pxl_max_horizontal
        self.pxl_max_vertical = pxl_max_vertical
        self.rand_sampler_horizontal = torch.distributions.uniform.Uniform(0, self.pxl_max_horizontal)
        self.rand_sampler_vertical = torch.distributions.uniform.Uniform(0, self.pxl_max_vertical)

    def __call__(self, img):
        w = img.shape[1]
        pxls_to_add_horizontal_left = int(self.rand_sampler_horizontal.sample().item())
        pxls_to_add_horizontal_right = int(self.rand_sampler_horizontal.sample().item())
        pxls_to_add_vertical_top = int(self.rand_sampler_vertical.sample().item())
        pxls_to_add_vertical_bottom = int(self.rand_sampler_vertical.sample().item())

        return cv2.copyMakeBorder(img, top=pxls_to_add_vertical_top, bottom=pxls_to_add_vertical_bottom, left=pxls_to_add_horizontal_left, right=pxls_to_add_horizontal_right, borderType=cv2.BORDER_REPLICATE)


class CropHorizontal(object):
    def __init__(self, pct):
        self.pct = pct
        self.rand_sampler = torch.distributions.uniform.Uniform(0, self.pct)

    def __call__(self, img):
        w = img.shape[1]
        pct = self.rand_sampler.sample().item()
        chopped_pixels = int(w*pct)

        if chopped_pixels == 0:
            return img
        else:
            return img[:, chopped_pixels:-chopped_pixels].copy()

class CropVertical(object):
    def __init__(self, pct):
        self.pct = pct
        self.rand_sampler = torch.distributions.uniform.Uniform(0, self.pct)

    def __call__(self, img):
        h = img.shape[0]
        pct = self.rand_sampler.sample().item()
        chopped_pixels = int(h*pct)

        if chopped_pixels == 0:
            return img
        else:
            return img[chopped_pixels:-chopped_pixels].copy()


class RandomCrop(object):
    """Crops the given opencv image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        if self.padding > 0:
            img = copyMakeBorder(img, top=self.padding, bottom=self.padding, left=self.padding, right=self.padding, borderType=cv2.BORDER_CONSTANT, value=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given opencv image with a probability of 0.5
    """

    def __call__(self, img):
        if random.random() < 0.5:
            return cv2.flip(img, flipCode=1)
            #return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class HorizontalFlip(object):
    """Randomly horizontally flips the given opencv image with a probability of 0.5
    """

    def __call__(self, img):
        return cv2.flip(img, flipCode=1)

class RandomSizedCrop(object):
    """Random crop the given opencv image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: BICUBIC
    """

    def __init__(self, size, interpolation=cv2.INTER_CUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))
