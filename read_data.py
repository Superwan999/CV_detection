import xml.dom.minidom as xmldom
import os

input_annotations_dir = ""  # the path of annotation data
output_labels_dir = "" # the path of labels

anno_list = os.listdir(input_annotations_dir)
for anno in anno_list:
    print(anno)
    anno_path = os.path.join(input_annotations_dir, anno)
    domobj = xmldom.parse(anno_path)
    elementobj = domobj.documentElement

    filename = elementobj.getElementsByTagName("filename")[0].firstChild.data
    img_W = int(elementobj.getElementsByTagName("width")[0].firstChild.data)
    img_H = int(elementobj.getElementsByTagName("height")[0].firstChild.data)

    save_txt_path = os.path.join(output_labels_dir, filename.split('.')[0] + 'txt')
    with open(save_txt_path, 'w') as f:
        objects = elementobj.getElementsByTagName("object")
        for object in objects:
            label_name = object.getElementsByTagName("name")[0].firstChild.data
            bnbox = object.getElementsByTagName("bndbox")[0]
            xmin = int(bnbox.getElementsByTagName("xmin")[0].firstChild.data)
            ymin = int(bnbox.getElementsByTagName("ymin")[0].firstChild.data)
            xmax = int(bnbox.getElementsByTagName("xmax")[0].firstChild.data)
            ymax = int(bnbox.getElementsByTagName("ymax")[0].firstChild.data)

            label_id = '0' if label_name == 'face' else 'l'
            x_center = str((xmin + xmax) / 2 / img_W)
            y_center = str((ymin + ymax) / 2 / img_H)
            width = str((xmax - xmin) / img_W)
            height = str((ymax - ymin) / img_H)

            data = label_id + ' ' + x_center + ' ' + y_center + ' ' + width + ' ' + height + '\n'
            f.write(data)


