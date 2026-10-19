"""
Inference code to lip-sync videos in the wild using Wav2Lip models.

This script takes a face video/image and an audio file, and synchronizes the lip movements
of the face to match the audio using the Wav2Lip model.
"""

import argparse
import math
import os
import shutil
import subprocess
import warnings
from typing import Any, Generator, List, Tuple

import audio
import cv2
import face_detection
import numpy as np
import torch
from models import Wav2Lip
from tqdm import tqdm

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore")

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
MEL_STEP_SIZE = 16
MEL_STEPS_PER_SECOND = 80.


def _str2bool(value: Any) -> bool:
    """argparse helper so --static False parses correctly (plain type=bool treats any non-empty string as truthy)."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value!r}")


def _is_image(path: str) -> bool:
    """True if the path has a still-image extension (case-insensitive)."""
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def run_ffmpeg(args: List[str]) -> None:
    """
    Runs ffmpeg as an argument list (never via shell=True) and raises on failure.

    Args:
        args (List[str]): ffmpeg arguments, excluding the "ffmpeg" executable itself.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install it and make sure it is accessible "
            "from this terminal (see README.md)."
        )

    cmd = ["ffmpeg", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit code {result.returncode}) running: {' '.join(cmd)}\n"
            f"{result.stderr.strip()}"
        )


def get_smoothened_boxes(boxes: np.ndarray, T: int) -> np.ndarray:
    """
    Smoothens the bounding boxes over a temporal window T to check for jitter/noise.

    Args:
        boxes (np.ndarray): Array of bounding boxes.
        T (int): Window size.

    Returns:
        np.ndarray: Smoothened bounding boxes.
    """
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[max(0, len(boxes) - T):]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def _is_out_of_memory(exc: BaseException) -> bool:
    """True if a RuntimeError looks like a GPU/CPU allocation failure rather than some unrelated bug."""
    message = str(exc).lower()
    return "out of memory" in message or "can't allocate memory" in message


def face_detect(images: List[np.ndarray], device: str, batch_size: int, pads: List[int], nosmooth: bool) -> List[List[Any]]:
    """
    Detects faces in a list of images using the specified device and batch size.

    Args:
        images (List[np.ndarray]): List of images (frames).
        device (str): 'cuda' or 'cpu'.
        batch_size (int): Batch size for face detection.
        pads (List[int]): Padding [top, bottom, left, right].
        nosmooth (bool): If True, disables smoothing.

    Returns:
        List[List[Any]]: List of results containing crop images and coordinates.
    """
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    while True:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size), desc="Face Detection"):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError as exc:
            # Only retry on genuine out-of-memory errors; anything else is a real bug and shouldn't be masked.
            if not _is_out_of_memory(exc):
                raise
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection. Please use the --resize_factor argument') from exc
            batch_size //= 2
            if device == 'cuda':
                torch.cuda.empty_cache()
            print(f'Recovering from OOM error; New batch size: {batch_size}')
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads

    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')

    for rect, image in zip(predictions, images):
        if rect is None:
            os.makedirs(temp_dir, exist_ok=True)
            cv2.imwrite(os.path.join(temp_dir, 'faulty_frame.jpg'), image)
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)

    final_results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    if device == 'cuda':
        torch.cuda.empty_cache()
    return final_results


def _stack_batch(img_batch: List[np.ndarray], mel_batch: List[np.ndarray], img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Builds the model inputs for one batch: (masked face + reference face) channels, and mels with a channel axis."""
    img_batch_arr = np.asarray(img_batch)
    mel_batch_arr = np.asarray(mel_batch)

    img_masked = img_batch_arr.copy()
    img_masked[:, img_size // 2:] = 0

    img_batch_arr = np.concatenate((img_masked, img_batch_arr), axis=3).astype(np.float32) / 255.
    mel_batch_arr = np.reshape(mel_batch_arr, [len(mel_batch), mel_batch_arr.shape[1], mel_batch_arr.shape[2], 1])
    return img_batch_arr, mel_batch_arr


def datagen(frames: List[np.ndarray], mels: List[np.ndarray], args: argparse.Namespace, device: str) -> Generator:
    """
    Generator function that yields batches of data for the model.
    """
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    # In static mode only the first frame is ever used, so don't detect faces in (or crop) the rest.
    det_frames = frames[:1] if args.static else frames

    if args.box[0] == -1:
        face_det_results = face_detect(det_frames, device, args.face_det_batch_size, args.pads, args.nosmooth)
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in det_frames]

    # The same face crop is fed to the model for every audio step in static mode: resize it once.
    static_face = None
    if args.static:
        static_face = cv2.resize(face_det_results[0][0], (args.img_size, args.img_size))

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        coords = face_det_results[idx][1]

        if static_face is not None:
            face = static_face
        else:
            face = cv2.resize(face_det_results[idx][0], (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch_arr, mel_batch_arr = _stack_batch(img_batch, mel_batch, args.img_size)
            yield img_batch_arr, mel_batch_arr, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch_arr, mel_batch_arr = _stack_batch(img_batch, mel_batch, args.img_size)
        yield img_batch_arr, mel_batch_arr, frame_batch, coords_batch


def _load(checkpoint_path: str, device: str) -> Any:
    """Loads a checkpoint, preferring PyTorch's safer weights_only mode when the installed torch version supports it."""
    map_location = None if device == 'cuda' else lambda storage, loc: storage
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older torch releases (<1.13) don't accept the weights_only kwarg at all.
        return torch.load(checkpoint_path, map_location=map_location)
    except Exception:
        # Non-tensor checkpoint contents rejected weights_only; only fall back for checkpoints you trust.
        print("[WARN] weights_only load failed; retrying with weights_only=False. "
              "Only do this for checkpoints you trust.")
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def load_model(path: str, device: str) -> Any:
    model = Wav2Lip()
    print(f"Load checkpoint from: {path}")
    checkpoint = _load(path, device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a Wav2Lip checkpoint (no 'state_dict' entry found).")
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def get_video_fps(path: str, default_fps: float) -> float:
    """Returns the video's frame rate, falling back to default_fps when the container doesn't report a usable one."""
    video_stream = cv2.VideoCapture(path)
    if not video_stream.isOpened():
        video_stream.release()
        raise ValueError(f'Could not open video: {path}')
    # An audio-only file "opens" fine but reports 0x0 and then blocks forever on read(), so reject it here.
    has_video = video_stream.get(cv2.CAP_PROP_FRAME_WIDTH) > 0 and video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT) > 0
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    video_stream.release()
    if not has_video:
        raise ValueError(f'{path} has no video stream; --face must be an image or a video file.')

    if not fps or not math.isfinite(fps) or fps <= 0:
        print(f"[WARN] Could not read a frame rate from {path}; assuming {default_fps} fps.")
        return default_fps
    return fps


def read_frames(args: argparse.Namespace, max_frames: int) -> List[np.ndarray]:
    """
    Reads (and resizes / rotates / crops) frames from the face image or video, stopping after max_frames
    so a long video paired with short audio doesn't load frames that would just be thrown away.
    """
    if _is_image(args.face):
        # imdecode(fromfile) rather than imread so non-ASCII paths work on Windows.
        image = cv2.imdecode(np.fromfile(args.face, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Could not decode image: {args.face}')
        if args.resize_factor > 1:
            image = cv2.resize(image, (max(1, image.shape[1] // args.resize_factor),
                                       max(1, image.shape[0] // args.resize_factor)))
        return [image]

    video_stream = cv2.VideoCapture(args.face)
    if not video_stream.isOpened():
        video_stream.release()
        raise ValueError(f'Could not open video: {args.face}')

    print('Reading video frames...')
    full_frames = []
    while len(full_frames) < max_frames:
        still_reading, frame = video_stream.read()
        if not still_reading:
            break

        if args.resize_factor > 1:
            frame = cv2.resize(frame, (max(1, frame.shape[1] // args.resize_factor),
                                       max(1, frame.shape[0] // args.resize_factor)))

        if args.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        y1, y2, x1, x2 = args.crop
        if x2 == -1:
            x2 = frame.shape[1]
        if y2 == -1:
            y2 = frame.shape[0]

        frame = frame[y1:y2, x1:x2]
        full_frames.append(frame)

    video_stream.release()
    return full_frames


def build_mel_chunks(mel: np.ndarray, fps: float) -> List[np.ndarray]:
    """Slices the mel spectrogram into one MEL_STEP_SIZE-wide window per output video frame."""
    # Very short clips (< ~0.2s) have fewer steps than one window; repeat the last step so one frame can still be made.
    if mel.shape[1] < MEL_STEP_SIZE:
        mel = np.pad(mel, ((0, 0), (0, MEL_STEP_SIZE - mel.shape[1])), mode='edge')

    mel_chunks = []
    mel_idx_multiplier = MEL_STEPS_PER_SECOND / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + MEL_STEP_SIZE > mel.shape[1]:
            mel_chunks.append(mel[:, mel.shape[1] - MEL_STEP_SIZE:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + MEL_STEP_SIZE])
        i += 1
    return mel_chunks


def open_video_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """Opens an intermediate AVI writer, trying several codecs since OpenCV builds differ in which they ship."""
    for codec in ('DIVX', 'MJPG', 'mp4v'):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError('Could not open an intermediate video writer (tried DIVX, MJPG, mp4v). '
                       'Your OpenCV build may be missing video codec support.')


def mux_audio_video(audio_path: str, video_path: str, outfile: str) -> None:
    """
    Combines the generated frames with the audio track into the final file.

    Prefers H.264/AAC at a sane quality with even dimensions (libx264 rejects odd widths/heights, which
    --resize_factor can easily produce) and falls back to ffmpeg's defaults if that encoder setup is unavailable.
    """
    try:
        run_ffmpeg(['-i', audio_path, '-i', video_path,
                    '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-shortest', outfile])
    except RuntimeError as exc:
        print(f"[WARN] H.264 encode failed, retrying with ffmpeg defaults:\n{exc}")
        run_ffmpeg(['-i', audio_path, '-i', video_path, '-strict', '-2', '-q:v', '1', outfile])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Name of saved checkpoint to load weights from')
    parser.add_argument('--face', type=str, required=True,
                        help='Filepath of video/image that contains faces to use')
    parser.add_argument('--audio', type=str, required=True,
                        help='Filepath of video/audio file to use as raw audio source')
    parser.add_argument('--outfile', type=str, default='results/result_voice.mp4',
                        help='Video path to save result. See default for an e.g.')

    parser.add_argument('--static', type=_str2bool, nargs='?', const=True, default=False,
                        help='If True, then use only first video frame for inference')
    parser.add_argument('--fps', type=float, default=25.,
                        help='Can be specified only if input is a static image (default: 25)')

    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')
    parser.add_argument('--face_det_batch_size', type=int, default=16,
                        help='Batch size for face detection')
    parser.add_argument('--wav2lip_batch_size', type=int, default=128,
                        help='Batch size for Wav2Lip model(s)')
    parser.add_argument('--resize_factor', default=1, type=int,
                        help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')
    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')
    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected. Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')
    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg. Use if you get a flipped result, despite feeding a normal looking video')
    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help="Force a compute device instead of auto-detecting CUDA availability")

    args = parser.parse_args()

    for name in ('pads', 'crop', 'box'):
        if len(getattr(args, name)) != 4:
            parser.error(f'--{name} expects exactly 4 integers (top, bottom, left, right)')
    for name in ('face_det_batch_size', 'wav2lip_batch_size', 'resize_factor'):
        if getattr(args, name) < 1:
            parser.error(f'--{name} must be at least 1')
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error('--fps must be a positive number')

    args.img_size = 96
    return args


def main():
    args = parse_args()

    # Establish device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('--device cuda was requested but CUDA is not available on this machine.')
        device = args.device
    print(f'Using {device} for inference.')

    # Ensure temp directory exists
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Check input types
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')
    if not os.path.isfile(args.audio):
        raise ValueError('--audio argument must be a valid path to an audio/video file')

    if _is_image(args.face):
        args.static = True
        fps = args.fps
    else:
        fps = get_video_fps(args.face, args.fps)

    # Handle Audio
    if not args.audio.lower().endswith('.wav'):
        print('Extracting raw audio...')
        temp_wav = os.path.join(temp_dir, 'temp.wav')
        run_ffmpeg(['-i', args.audio, '-strict', '-2', temp_wav])
        args.audio = temp_wav

    wav = audio.load_wav(args.audio, 16000)
    if wav.size == 0:
        raise ValueError(f'No audio samples could be read from {args.audio}')
    mel = audio.melspectrogram(wav)
    print(f"Mel Spectrogram Shape: {mel.shape}")

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    # Prepare Mel Chunks (one per output frame)
    mel_chunks = build_mel_chunks(mel, fps)
    print(f"Length of mel chunks: {len(mel_chunks)}")

    # Read only as many frames as will actually be used (a single one in static mode).
    full_frames = read_frames(args, 1 if args.static else len(mel_chunks))
    if not full_frames:
        raise ValueError(f'No frames could be read from {args.face}')
    print(f"Number of frames available for inference: {len(full_frames)}")

    batch_size = args.wav2lip_batch_size

    # Load the model up front so a bad checkpoint fails immediately, not after the slow face-detection pass.
    model = load_model(args.checkpoint_path, device)
    print("Model loaded")

    # Generate Batches
    gen = datagen(full_frames.copy(), mel_chunks, args, device)

    frame_h, frame_w = full_frames[0].shape[:2]
    temp_avi = os.path.join(temp_dir, 'result.avi')
    out = open_video_writer(temp_avi, fps, (frame_w, frame_h))

    try:
        for img_batch, mel_batch, frames, coords in tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)),
                                                          desc="Wav2Lip Processing"):
            img_batch = torch.from_numpy(np.ascontiguousarray(np.transpose(img_batch, (0, 3, 1, 2)))).to(device)
            mel_batch = torch.from_numpy(
                np.ascontiguousarray(np.transpose(mel_batch, (0, 3, 1, 2))).astype(np.float32)).to(device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)
    finally:
        out.release()

    # Combine Audio and Video
    outfile_dir = os.path.dirname(args.outfile)
    if outfile_dir:
        os.makedirs(outfile_dir, exist_ok=True)
    mux_audio_video(args.audio, temp_avi, args.outfile)
    print(f"Final output saved to: {args.outfile}")


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1) from exc
