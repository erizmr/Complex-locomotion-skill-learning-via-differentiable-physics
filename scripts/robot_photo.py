import glob
from PIL import Image

all_images = sorted(glob.glob("./robot_design_data/*.png"))[:-1] # the snake is not included here

print(all_images)
frame_num = len(all_images)

images = [Image.open(x) for x in all_images]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
    new_im.paste(im, (x_offset, 0))
    x_offset += im.size[0]
new_im.save('./robot_design_data/robot_photo.pdf')