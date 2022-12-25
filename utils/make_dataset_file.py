import os
import numpy as np

def make_list(dataset_name, file_root, dataset_dir):
	if not os.path.exists(file_root):
		os.makedirs(file_root)
	for domain in os.listdir(dataset_dir):
		file_name = '{}.txt'.format(domain)
		file_path = os.path.join(file_root, file_name)
		if dataset_name.lower() in ['office-31', 'office31']: domain += '/images'
		with open(file_path, 'a') as f:
			for idx, label in enumerate(os.listdir(os.path.join(dataset_dir, domain))):
				image_root = os.path.join(domain, label)
				for file in os.listdir(os.path.join(dataset_dir, image_root)):
					file_path = os.path.join(image_root, file).replace('\\', '/')
					f.write(file_path + ' ' + str(idx) + '\n')
		print('{} created!'.format(file_name))

dataset_name = 'pacs'
file_root = '/diskD/experiment/feature_extractor/for_image_data/code/data/list/{}'.format(dataset_name)
dataset_dir = '/diskD/experiment/feature_extractor/for_image_data/data/{}'.format(dataset_name)
