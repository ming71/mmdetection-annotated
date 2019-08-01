# Notes!!
MMDetection-annotations have been update to latest **version 1.0**. I'll continue finish the remaining part and may not chase after upgrades in the future (present version is good enough) 
# mmdetection-annotated 

## Introduction
Refer to the execllent implemention here:https://github.com/open-mmlab/mmdetection ,and thanks to author [Kai Chen](https://github.com/hellock).</br>
Open-mmlab project , which contains various models and implementions of latest papers , achieves great results in detection/segmentataion tasks , and is kind enough for rookies in CV field.</br>

## Getting started
More information about installation or pre-train model downloads , pls refer to [officia mmdetection](https://github.com/open-mmlab/mmdetection) or [blog here](https://blog.csdn.net/mingqi1996/article/details/88091802)</br>
* **Test on images</br>**
You can test on Faster RCNN demo by running the script `demo.py`.
I have just rewritten the demo file to detect on single image or a folder as follow:
```
import os
from mmdet.apis import init_detector, inference_detector, show_result

if __name__ == '__main__':
	config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
	checkpoint_file = 'weights/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
	# checkpoint_file = 'tools/work_dirs/mask_rcnn_r101_fpn_1x/epoch_1200.pth'
	img_path = '/home/bit/下载/n07753592'
	model = init_detector(config_file, checkpoint_file, device='cuda:0')
	# print(model)
	# 输入可以为文件夹或者图片
	if os.path.isdir(img_path):
		imgs= os.listdir(img_path)
		for i in range(len(imgs)):
			imgs[i]=os.path.join(img_path,imgs[i])
		for i, result in enumerate(inference_detector(model, imgs)):	# 支持可迭代输入imgs
			print(i, imgs[i])
			show_result(imgs[i], result, model.CLASSES, out_file='output/result_{}.jpg'.format(i))

	elif os.path.isfile(img_path):
		result = inference_detector(model, img_path)
		show_result(img_path, result, model.CLASSES)


```
* **Debug**  
You can debug by setting breakpoint with method of adding `ipdb.set_trace()`.Before that , make sure of the success installment and import of **ipdb** package.
* **Hook**  
If you want to inspect on intermediate variables , `hook.py` can be a provision served as a reference for your work.
## Annotations
Annotations are attached everywhere in the code(surely only the part I have read , and the not finished part will be completed as soon as possible). Beside , `annotation` folder contains some interpreting documents as well.  
* **Dataset Example**   
Provide a simple small sample data set for testing (segmentation && detection) .More details referrd to instruction [here](https://blog.csdn.net/mingqi1996/article/details/96706619)

* **CUDA related code**  
I've delete files in folder mmdet/ops cause no annotations attached inside.However it's a good news that specific notes are made about RoIAlign [here](https://zhuanlan.zhihu.com/p/75171514) .

* **Model visualization**  
Take Mask-RCNN for example , the model can be visualized as follow:(more details refere to [model-structure-png](https://github.com/ming71/mmdetection-annotated/blob/master/annotation/model_vis/maskrcnn-model-inference.png))
<div align=center><img src="https://github.com/ming71/mmdetection-annotated/blob/master/annotation/model_vis/inference.png"/></div>

* **Configuration**  
Explicit describtion on config file , take Mask RCNN for example , refer to [mask_rcnn_r101_fpn_1x.py](https://github.com/ming71/mmdetection-annotated/blob/master/annotation/mask_rcnn_r101_fpn_1x.py)  

* **MMCV&MMDET**  
Specification of mmcv lib and a partial of mmdet(more details about various models will be updated later ).</br>

## Detection Results</br>
Test on Mask RCNN model:  
<div align=center><img src="https://github.com/ming71/mmdetection-annotated/blob/master/outputs/_s1019.png"/></div>
<div align=center><img  src="https://github.com/ming71/mmdetection-annotated/blob/master/outputs/_screenshot_02.04.2019.png"/></div>
<div align=center><img  src="https://github.com/ming71/mmdetection-annotated/blob/master/outputs/_screenshot_071019.png"/></div>


## Training</br>
### **dataset**<br>
- You can just use COCO dataset , refer [here](https://blog.csdn.net/mingqi1996/article/details/88091802).<br>
- If you want to train on your customed dataset labeled by `labelme` , you need first convert json files to COCO style , this [toolbox](https://github.com/ming71/toolbox) may help you ;<br>
- If you want to train on your customed dataset labeled by `labelImg` , you need first convert xml files to COCO style , this [toolbox](https://github.com/ming71/toolbox) may also help you .<br>
- I have tested on these tools recently to make sure them still work well, if questiones still arised , desrcibe on issue please or contact me , thanks.<br>

### learning rate
Remember to set lr in config file according to your <u>**own GPU_NUM**</u> !!!!(eg.1/8 of default lr for 1 GPU)

## Future work</br>
Mmdetection performs better than many classical implementions , it's really a excellent work , can be called as ‘Chinese Detectron’ :p . I will update this project with annotations for more details in the future, letting more people make a good use of this great work.You can continue to foucus on this repo.</br>
BTW , this repo is just used for better comprehension , if you ask for better performance or latest paper implementions ,please keep eyes on [mmdetection](https://github.com/open-mmlab/mmdetection)</br>

