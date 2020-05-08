label = [0.0438272, 0.63287, 0.0851852, 0.402778]
cls = 0
with open('/home/orcun/yolov3/output/bus' + '.txt', 'a') as file:
    file.write(('%g ' * 4 + '%g' + '\n') % (cls, *label))