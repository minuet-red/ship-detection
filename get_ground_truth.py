# ----------------------------------------------------------------------------------------------------------------------
# 将xml文件转成符合要求的txt文件
# ----------------------------------------------------------------------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import cv2

# ----------------------------------------------------------------------------------------------------------------------

l = os.listdir("data/annotations/")
l.sort()
out = []
for i in range(0, len(l)):
    if l[i].endswith("xml"):
        out.append(l[i])

# ----------------------------------------------------------------------------------------------------------------------

for j in range(0, len(out)):
    root = ET.parse("data/annotations/" + out[j]).getroot()
    object_num = len(root.getchildren())
    name = out[j].split(".")[0]
    output = open("data/ground-truth/" + name + ".txt", "a")

    for k in range(0, object_num - 5):
        children_node = root.getchildren()[5 + k]

        length = len(list(children_node))
        location = []
        for i in range(0, length):
            if i == 0:
                CLASS = list(children_node)[i].text.strip()

            if i == 4:
                for item in list(children_node)[i]:
                    location.append(item.text.strip())

                output.write(CLASS + " "
                             + str(location[0]) + " "
                             + str(location[1]) + " "
                             + str(location[2]) + " "
                             + str(location[3]) + "\n")


# ----------------------------------------------------------------------------------------------------------------------




