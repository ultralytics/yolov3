
import glob
lines = glob.glob("data_xview/train/images/*")
with open("data_xview/xview_img.txt", mode='w', encoding='utf-8') as myfile:
        myfile.write('\n'.join(lines))
