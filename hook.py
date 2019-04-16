import mmcv
import torch
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import ipdb

def roialign_forward(module,input,output):
	print('\n\ninput:')
	print(input[0].shape,'\n',input[1].shape)
	print('\n\noutput:')
	print(output.shape)
	# print(type(input))


if __name__ == '__main__':
	params=[]
	def hook(module,input):
		# print('breakpoint')
		params.append(input)
		# print(input[0].shape)
		# data=input
	cfg = mmcv.Config.fromfile('configs/faster_rcnn_r50_fpn_1x.py')
	cfg.model.pretrained = None

	torch.cuda.empty_cache()

	# ipdb.set_trace()

	# construct the model and load checkpoint
	model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
	print(model)
	handle=model.backbone.conv1.register_forward_pre_hook(hook)
	# model.bbox_roi_extractor.roi_layers[0].register_forward_hook(roialign_forward)
	
	_ = load_checkpoint(model, 'weights/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')
	
	# test a single image
	img= mmcv.imread('/py/pic/2.jpg')
	result = inference_detector(model, img, cfg)
	# print(params)

	
	show_result(img, result)
	handle.remove()
	# # test a list of images
	# imgs = ['/py/pic/4.jpg', '/py/pic/5.jpg']
	# for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
	#     print(i, imgs[i])
	#     show_result(imgs[i], result)

