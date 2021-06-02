import imageio
import numpy as np
from pathlib import Path
from PIL import Image
import cv2


filename = Path('/Users/arminbahl/Dropbox/mutant_manuscript/paper/BioRXiv submission March 2021/videoS2.mp4')
vid = imageio.get_reader(filename,  'ffmpeg')
input_fps = vid.get_meta_data()['fps']
print(input_fps)

writer = imageio.get_writer(str(filename.parent / filename.stem) + "_with_circle.mp4", fps=input_fps, ffmpeg_params=["-b:v", "4M"])

for i, frame in enumerate(vid):
    frame = np.array(Image.fromarray(frame))

    cv2.circle(frame, (512, 512), 512, (255, 255, 255), 2)

    writer.append_data(frame)

writer.close()