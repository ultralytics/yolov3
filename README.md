# 参考
项目Fork于 https://github.com/ultralytics/yolov3  
参考blog:https://blog.csdn.net/public669/article/details/98020800  

# 快速使用
1. git clone https://github.com/zdyshine/yolov3.git  
2. data文件夹下新建 Annotations, images, ImageSets, labels文件夹（已建）  
3. 把图片*.jpg放置 data/ImageSets， 把*.xml文件放置到data/Annotations  
4. data文件中新建 under_water.data, underwater_names文件（已建）  
5. 分别运行makeTxt.py和voc_label.py创建训练需要的文件  
6. 修改yolo-tiny.cfg，原理参考上面的blog  
7. 根据训练需要，修改train.py中的训练参数开始训练，模型保存在weight/  
8. 修改detect_underwater.py测试数据路径和模型路径开始测试  
9. 结果保存在当前目录submit.csv  
注：步骤2~7主要参考blog，请仔细阅读该blog步骤。  

