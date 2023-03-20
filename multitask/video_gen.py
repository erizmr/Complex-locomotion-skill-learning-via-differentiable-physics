import os
os.system("cd video\nrm *.gif")
dir_list = os.listdir("video/")
for name in dir_list:
    os.system("cd video/{}\nti video && ti gif -i video.mp4 -f250 && mv video.gif ../{}.gif".format(name, name))
