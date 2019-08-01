from .voc import VOCDataset
from .registry import DATASETS


@DATASETS.register_module
class MyDataset(VOCDataset):

	CLASSES = ('large-vehicle', 'swimming-pool', 'helicopter', 'bridge', 'plane','ship',
				'soccer-ball-field','basketball-court','airport','container-crane',
				'ground-track-field','small-vehicle','harbor','baseball-diamond','tennis-court',
				'roundabout','storage-tank','helipad')

	# CLASSES = ('ship','cruiser','carrier')


















