
import os
from mmdet.apis import init_detector, inference_detector, show_result

if __name__ == '__main__':
	config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
	checkpoint_file = 'weights/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
	# checkpoint_file = 'tools/work_dirs/mask_rcnn_r101_fpn_1x/epoch_1200.pth'
	img_path = '/home/bit/下载/n07753592'

	model = init_detector(config_file, checkpoint_file, device='cuda:0')

	# print(model)

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

