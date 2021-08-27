import sys
import os
import glob
from PIL import Image

import argparse
parser = argparse.ArgumentParser(description='draw')
parser.add_argument('--robot_id',
                    default='',
                    help='robot id to compare')
args = parser.parse_args()
all_folders = sorted(glob.glob(f"./video/interactive/robot_{args.robot_id}/*"))#, key=os.path.getmtime)
os.makedirs(f'./video/compare_results/robot_{args.robot_id}', exist_ok=True)

print(all_folders)
all_images = []
for p in all_folders:
    imgs = sorted(glob.glob(p + "/*.png"))
    all_images.append(imgs)
# print(all_images)
frame_num = len(all_images[0])
print(frame_num)

for n in range(frame_num):
    images = [Image.open(x[n]) for x in all_images]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    new_im.save('./video/compare_results/robot_{}/{:04d}.png'.format(args.robot_id, n))