import os
dir_list = os.listdir("mass_spring/")
os.system("cd mass_spring\nrm *.gif")
for name in dir_list:
    os.system("cd mass_spring/{}\nti video && ti gif -i video.mp4 -f250 && mv video.gif ../{}.gif".format(name, name))
