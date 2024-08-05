try:
    import decord
    from decord import VideoReader, gpu, cpu
except ImportError as e:
    decord = None
    print(e)

import numpy as np


def read_video_decord(video_path, num_frames):
    """
    Decode the video with Decord decoder.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample uniformly. Defaults to NUM_FRAMES

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    """
    vr = VideoReader(
        uri=video_path, ctx=cpu(0)
    )  # you need to install from source to use gpu ctx
    indices = np.arange(0, len(vr), len(vr) / num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()
    return frames


def get_video_reader():
    if decord:
        return read_video_decord
    else:
        raise NotImplementedError
