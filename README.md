①`python calcOpticalFlow.py test`

用于测试光流计算函数，对tsukuba_l.png和tsukuba_r.png两张图片进行计算。该模式将先显示各像素点位移大小的统计直方图， 然后显示tusukuba_l.png上光流较大区域的光流。

按Esc键结束程序。

② `python calcOpticalFlow.py eval-data`

​    用于处理文件夹eval-data中的所有图片，并将结果保存在‘result’文件夹中。

③`python calcOpticalFlow.py video filename`

​    用于处理视频文件filename，并将结果保存成‘result.avi’

④ `python calcOpticalFlow.py camera`

​    用于处理摄像头实时图像，并将结果保存成‘camera.avi’

​    按Space键开始或者暂停录制，按Esc键退出程序。