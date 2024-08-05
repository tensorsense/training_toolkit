from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np


def animate_video_sample(sample):
    # convert to image from proceessed tensors
    clip = sample["pixel_values_videos"][0] * 255
    clip = clip.permute(0, 2, 3, 1).clamp(0, 255)

    # np array with shape (frames, height, width, channels)
    video = np.array(clip).astype(np.uint8)

    fig = plt.figure()
    im = plt.imshow(video[0, :, :, :])

    plt.close()  # this is required to not display the generated image

    def init():
        im.set_data(video[0, :, :, :])

    def animate(i):
        im.set_data(video[i, :, :, :])
        return im

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=video.shape[0], interval=100
    )

    return anim
