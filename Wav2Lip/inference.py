"""
Inference code to lip-sync videos in the wild using Wav2Lip models.

This script takes a face video/image and an audio file, and synchronizes the lip movements
of the face to match the audio using the Wav2Lip model.
"""

import argparse
import os
import platform
import subprocess

import warnings

from typing import List, Generator, Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

import audio
import face_detection
from models import Wav2Lip

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore")


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
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


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
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print(f'Recovering from OOM error; New batch size: {batch_size}')
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    
    # We define TEMP_DIR globally later or pass it in. Relying on global for now as per original design but cleaner.
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')

    for rect, image in zip(predictions, images):
        if rect is None:
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
    return final_results


def datagen(frames: List[np.ndarray], mels: List[np.ndarray], args: argparse.Namespace, device: str) -> Generator:
    """
    Generator function that yields batches of data for the model.
    """
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames, device, args.face_det_batch_size, args.pads, args.nosmooth) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]], device, args.face_det_batch_size, args.pads, args.nosmooth)
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch_arr = np.asarray(img_batch)
            mel_batch_arr = np.asarray(mel_batch)

            img_masked = img_batch_arr.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch_arr = np.concatenate((img_masked, img_batch_arr), axis=3) / 255.
            mel_batch_arr = np.reshape(mel_batch_arr, [len(mel_batch), mel_batch_arr.shape[1], mel_batch_arr.shape[2], 1])

            yield img_batch_arr, mel_batch_arr, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch_arr = np.asarray(img_batch)
        mel_batch_arr = np.asarray(mel_batch)

        img_masked = img_batch_arr.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch_arr = np.concatenate((img_masked, img_batch_arr), axis=3) / 255.
        mel_batch_arr = np.reshape(mel_batch_arr, [len(mel_batch), mel_batch_arr.shape[1], mel_batch_arr.shape[2], 1])

        yield img_batch_arr, mel_batch_arr, frame_batch, coords_batch


def _load(checkpoint_path: str, device: str) -> Any:
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path: str, device: str) -> Any:
    model = Wav2Lip()
    print(f"Load checkpoint from: {path}")
    checkpoint = _load(path, device)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Name of saved checkpoint to load weights from')
    parser.add_argument('--face', type=str, required=True, 
                        help='Filepath of video/image that contains faces to use')
    parser.add_argument('--audio', type=str, required=True, 
                        help='Filepath of video/audio file to use as raw audio source')
    parser.add_argument('--outfile', type=str, default='results/result_voice.mp4', 
                        help='Video path to save result. See default for an e.g.')

    parser.add_argument('--static', type=bool, default=False, 
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

    args = parser.parse_args()
    args.img_size = 96
    
    # Establish device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} for inference.')

    # Ensure temp directory exists
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Check input types
    if os.path.isfile(args.face) and args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        args.static = True

    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    # Read frames
    if args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
        if args.resize_factor > 1:
            full_frames[0] = cv2.resize(full_frames[0], 
                                    (full_frames[0].shape[1]//args.resize_factor, full_frames[0].shape[0]//args.resize_factor))
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print(f"Number of frames available for inference: {len(full_frames)}")

    # Handle Audio
    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        temp_wav = os.path.join(temp_dir, 'temp.wav')
        command = f'ffmpeg -y -i "{args.audio}" -strict -2 "{temp_wav}"'
        subprocess.call(command, shell=True)
        args.audio = temp_wav

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(f"Mel Spectrogram Shape: {mel.shape}")

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    # Prepare Mel Chunks
    mel_step_size = 16
    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print(f"Length of mel chunks: {len(mel_chunks)}")

    full_frames = full_frames[:len(mel_chunks)]
    batch_size = args.wav2lip_batch_size
    
    # Generate Batches
    gen = datagen(full_frames.copy(), mel_chunks, args, device)

    # Model and Writer placeholders
    model = None
    out = None

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                            total=int(np.ceil(float(len(mel_chunks))/batch_size)), desc="Wav2Lip Processing")):
        if i == 0:
            model = load_model(args.checkpoint_path, device)
            print("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            temp_avi = os.path.join(temp_dir, 'result.avi')
            out = cv2.VideoWriter(temp_avi, 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    if out:
        out.release()
    else:
        print("[ERROR] No output generated.")
        return

    # Combine Audio and Video
    temp_avi = os.path.join(temp_dir, 'result.avi')
    command = f'ffmpeg -y -i "{args.audio}" -i "{temp_avi}" -strict -2 -q:v 1 "{args.outfile}"'
    subprocess.call(command, shell=platform.system() != 'Windows')
    print(f"Final output saved to: {args.outfile}")


if __name__ == '__main__':
    main()
