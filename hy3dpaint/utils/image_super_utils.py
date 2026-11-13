# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import numpy as np
import torch
from PIL import Image


class imageSuperNet:
    """RealESRGAN x4, loaded with spandrel instead of realesrgan/basicsr (basicsr no longer builds from source on
    recent Python and breaks on recent torchvision). Same network and weights, and the same steps as
    RealESRGANer.enhance with tile=0, pre_pad=0 and half=True, including its BGR channel order."""

    def __init__(self, config) -> None:
        from spandrel import ModelLoader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelLoader().load_from_file(config.realesrgan_ckpt_path).model.eval().half().to(self.device)

    @torch.no_grad()
    def __call__(self, image):
        img = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))[None]
        out = self.model(x.to(self.device).half()).float().clamp_(0, 1)[0].cpu().numpy()
        out = out[::-1].transpose(1, 2, 0)
        return Image.fromarray((out * 255.0).round().astype(np.uint8))
