"""The vectorized S3FD batch_detect must produce the same detections (after NMS) as the original per-anchor loop."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from face_detection.detection.sfd import detect as detect_mod
from face_detection.detection.sfd.bbox import batch_decode, nms
from face_detection.detection.sfd.net_s3fd import s3fd


def reference_batch_detect(net, imgs, device):
    """The original (slow) implementation, kept verbatim as the oracle."""
    imgs = imgs - np.array([104, 117, 123])
    imgs = imgs.transpose(0, 3, 1, 2)
    imgs = torch.from_numpy(imgs).float().to(device)
    BB, CC, HH, WW = imgs.size()
    with torch.no_grad():
        olist = net(imgs)

    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)
    olist = [oelem.data.cpu() for oelem in olist]
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        stride = 2**(i + 2)
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            score = ocls[:, 1, hindex, windex]
            loc = oreg[:, :, hindex, windex].contiguous().view(BB, 1, 4)
            priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]).view(1, 1, 4)
            box = batch_decode(loc, priors, [0.1, 0.2])
            box = box[:, 0] * 1.0
            bboxlist.append(torch.cat([box, score.unsqueeze(1)], 1).cpu().numpy())
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, BB, 5))
    return bboxlist


def after_nms(bboxlist):
    """What the detector does with raw output: per-image NMS, then the 0.5 score cut. Sorted for comparison."""
    per_image = []
    for i in range(bboxlist.shape[1]):
        dets = bboxlist[:, i, :]
        dets = dets[nms(dets, 0.3), :]
        dets = dets[dets[:, -1] > 0.5]
        per_image.append(dets[np.lexsort((dets[:, 0], dets[:, 4]))])
    return per_image


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return s3fd().eval()


@pytest.mark.parametrize("shape", [(2, 64, 64, 3), (3, 96, 128, 3), (1, 33, 47, 3)])
def test_vectorized_matches_reference(net, shape):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 255, shape).astype(np.uint8)

    expected = after_nms(reference_batch_detect(net, imgs, "cpu"))
    actual = after_nms(detect_mod.batch_detect(net, imgs, "cpu"))

    assert len(expected) == len(actual) == shape[0]
    for want, got in zip(expected, actual):
        assert want.shape == got.shape
        assert len(got) > 0  # random weights should still give the comparison something to chew on
        np.testing.assert_allclose(got, want, atol=1e-4)


def test_no_candidates_returns_sentinel():
    class NothingFires(torch.nn.Module):
        def forward(self, x):
            outs = []
            for _ in range(6):
                cls = torch.zeros(x.shape[0], 2, 4, 4)
                cls[:, 0] = 10  # background class dominates -> class-1 probability ~0 everywhere
                outs += [cls, torch.zeros(x.shape[0], 4, 4, 4)]
            return outs

    result = detect_mod.batch_detect(NothingFires(), np.zeros((2, 16, 16, 3), np.uint8), "cpu")
    assert result.shape == (1, 2, 5)
    assert after_nms(result)[0].shape[0] == 0
