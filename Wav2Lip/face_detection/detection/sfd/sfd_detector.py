import os
import torch
from torch.utils.model_zoo import load_url

from ..core import FaceDetector

from .net_s3fd import s3fd
from .bbox import *
from .detect import *

models_urls = {
    's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
}


class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=None, verbose=False):
        super(SFDDetector, self).__init__(device, verbose)

        # S3FD_MODEL_PATH lets the weights live elsewhere (and lets tests substitute their own).
        if path_to_detector is None:
            path_to_detector = os.environ.get('S3FD_MODEL_PATH') or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 's3fd.pth')

        # Initialise the face detector
        if not os.path.isfile(path_to_detector):
            model_weights = load_url(models_urls['s3fd'], map_location='cpu')
        else:
            try:
                # Prefer the safer weights_only mode on torch versions that support it.
                # map_location='cpu' so weights saved on a GPU machine still load on CPU-only ones.
                model_weights = torch.load(path_to_detector, map_location='cpu', weights_only=True)
            except TypeError:
                model_weights = torch.load(path_to_detector, map_location='cpu')

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image, device=self.device)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    def detect_from_batch(self, images):
        bboxlists = batch_detect(self.face_detector, images, device=self.device)
        keeps = [nms(bboxlists[:, i, :], 0.3) for i in range(bboxlists.shape[1])]
        bboxlists = [bboxlists[keep, i, :] for i, keep in enumerate(keeps)]
        bboxlists = [[x for x in bboxlist if x[-1] > 0.5] for bboxlist in bboxlists]

        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
