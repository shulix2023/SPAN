import torch
import torchvision.transforms as transforms
import os.path
from PIL import Image
import random
import torch.nn.functional as F
import xml.dom.minidom


def f_transforms(resize=256, isvgg=False):
	if isvgg:
		norm_val1, norm_val2 = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
		norm_val1_a, morm_val2_a = [0.485, 0.456, 0.406, 0], [0.229, 0.224, 0.225, 1]
	else:
		norm_val1, norm_val2 = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
		norm_val1_a, morm_val2_a = (0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)
	data_transforms = {
		'train': transforms.Compose([
			transforms.Resize((resize, resize)),
			transforms.ToTensor(),
			transforms.Normalize(norm_val1, norm_val2)
		]),
		'test': transforms.Compose([
			transforms.Resize((resize, resize)),
			transforms.ToTensor(),
			transforms.Normalize(norm_val1, norm_val2)
		]),
		'RGBA': transforms.Compose([
			transforms.Resize((resize, resize)),
			transforms.ToTensor(),
			transforms.Normalize(norm_val1_a, morm_val2_a)
		]),
		'mask': transforms.Compose([
			transforms.Resize((resize, resize)),
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))
		]),
		'xml': transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5*resize,), (0.5*resize,))
		])
	}
	return data_transforms


class CompDataset(object):
	def __init__(self, args, phase='train', lbl='h_lbl'):
		super(CompDataset, self).__init__()
		self.args = args
		self.root = os.path.join(args.root, phase)
		path_c = os.path.join(args.root, 'c')
		list_c = os.listdir(path_c)
		self.list_groups = []
		self.istrain = True

		if phase == 'test':
			self.istrain = False

		for c in list_c:
			if args.cup_list[0] > 0:
				if int(c.split('cup')[1].split('_')[0]) not in self.args.cup_list:
					continue
			full_path_c = os.path.join(args.root, 'c', c)
			cup = c.split('_')[0]
			if not os.path.exists(os.path.join(self.root, 'h', cup)):
				continue
			list_h_cup = sorted(os.listdir(os.path.join(self.root, 'h', cup)))
			list_n_cup = sorted(os.listdir(os.path.join(self.root, 'n', cup)))
			list_s_cup = sorted(os.listdir(os.path.join(self.root, 'seg', cup)))
			list_hxy_cup = sorted(os.listdir(os.path.join(self.root, lbl, cup)))

			for i in range(len(list_n_cup)):
				name_h = list_h_cup[i]
				name_n = list_n_cup[i]
				name_s = list_s_cup[i]
				name_hxy = list_hxy_cup[i]
				full_path_h = os.path.join(self.root, 'h', cup, list_h_cup[i])
				full_path_n = os.path.join(self.root, 'n', cup, list_n_cup[i])
				full_path_s = os.path.join(self.root, 'seg', cup, list_s_cup[i])

				full_path_hxy = os.path.join(self.root, lbl, cup, list_hxy_cup[i])

				if name_n[1:] == name_h[1:] and name_h[:-4] == name_hxy[:-4] and name_s[1:] == name_h[1:] :
					self.list_groups.append([full_path_c, full_path_n, full_path_h, full_path_s, full_path_hxy])
				else:
					print(
						f'Warning!!! The picture {full_path_h} is not match {full_path_n} or {full_path_hxy}!!!')

		self.size = len(self.list_groups)

	def __getitem__(self, index):
		n = index % self.size
		path_c = self.list_groups[n][0]
		path_n = self.list_groups[n][1]
		path_h = self.list_groups[n][2]
		path_s = self.list_groups[n][3]
		path_hxy = self.list_groups[n][4]

		img_c = Image.open(path_c).convert("RGBA")
		img_n = Image.open(path_n).convert("RGB").rotate(-90, expand=True)
		img_h = Image.open(path_h).convert("RGB").rotate(-90, expand=True)
		img_s = Image.open(path_s).convert("RGB")

		if self.istrain:
			self.read_xml(path_hxy)
			box_c = img_h.crop((self.xmin_r, self.ymin_r, self.xmax_r, self.ymax_r))
			img_n = img_n.crop((self.x1c, self.y1c, self.x2c, self.y2c))
			img_h = img_h.crop((self.x1c, self.y1c, self.x2c, self.y2c))
			img_s = img_s.crop((self.x1c, self.y1c, self.x2c, self.y2c))

			box_xy = torch.tensor([self.xmin_rrc, self.xmax_rrc, self.ymin_rrc, self.ymax_rrc], dtype=torch.int)
			box_xy_norm = (box_xy - self.args.resize / 2) / (self.args.resize / 2)
			box_xy_norm = box_xy_norm.to(dtype=torch.float)
		else:
			self.read_xml(path_hxy)
			box_c = img_h.crop((self.xmin_r, self.ymin_r, self.xmax_r, self.ymax_r))
			box_xy = torch.tensor([self.xmin_rr, self.xmax_rr, self.ymin_rr, self.ymax_rr], dtype=torch.int)
			box_xy_norm = (box_xy - self.args.resize / 2) / (self.args.resize / 2)
			box_xy_norm = box_xy_norm.to(dtype=torch.float)

		data_transforms = f_transforms(resize=self.args.resize)
		img_c = data_transforms['RGBA'](img_c)
		img_n = data_transforms['train'](img_n)
		img_h = data_transforms['train'](img_h)
		img_s = data_transforms['train'](img_s)
		box_c = data_transforms['train'](box_c)

		if random.random() < 0.5:
			scale = 0.2
			theta = torch.tensor([2.0, 0, 0, 0, 2.0, 0])
			dt1 = 2 * torch.rand_like(theta) - 1
			dt1[1:4] = 0
			dt1[-1] = 0
			dt2 = scale * torch.rand_like(theta) - scale / 2
			dt2[0] = 0
			dt2[-2] = 0
			theta = theta + dt1 + dt2
			theta = theta.view(-1, 2, 3)
			class_c = img_c[0:3].unsqueeze(0)
			grid = F.affine_grid(theta, class_c.size(), align_corners=False)
			class_c = F.grid_sample(class_c, grid, align_corners=False).squeeze(0)
		else:
			class_c = img_h

		return {'img_c': img_c, 'img_n': img_n, 'img_h': img_h, 'img_s': img_s, class_c: class_c,
				'box_c': box_c,	'path_c': path_c, 'path_n': path_n, 'path_h': path_h, 'path_s': path_s,
				'box_xy': box_xy, 'box_xy_norm': box_xy_norm}

	def __len__(self):
		return self.size

	def read_xml(self, path):
		dom = xml.dom.minidom.parse(path)
		rootdata = dom.documentElement
		itemlist = rootdata.getElementsByTagName('xmin')
		self.xmin = int(itemlist[0].firstChild.data)
		itemlist = rootdata.getElementsByTagName('ymin')
		self.ymin = int(itemlist[0].firstChild.data)
		itemlist = rootdata.getElementsByTagName('xmax')
		self.xmax = int(itemlist[0].firstChild.data)
		itemlist = rootdata.getElementsByTagName('ymax')
		self.ymax = int(itemlist[0].firstChild.data)
		itemlist = rootdata.getElementsByTagName('width')
		self.w = int(itemlist[0].firstChild.data)
		itemlist = rootdata.getElementsByTagName('height')
		self.h = int(itemlist[0].firstChild.data)

		x0 = -self.ymin + self.h
		x1 = -self.ymax + self.h
		self.xmin_r = min(x0, x1)
		self.xmax_r = max(x0, x1)
		y0 = self.xmin
		y1 = self.xmax
		self.ymin_r = min(y0, y1)
		self.ymax_r = max(y0, y1)
		self.h_r = self.w
		self.w_r = self.h

		self.xmin_rr = round(self.xmin_r / self.w_r * self.args.resize)
		self.xmax_rr = round(self.xmax_r / self.w_r * self.args.resize)
		self.ymin_rr = round(self.ymin_r / self.h_r * self.args.resize)
		self.ymax_rr = round(self.ymax_r / self.h_r * self.args.resize)
		self.x1c, self.y1c, self.x2c, self.y2c = self.crop_xy_ratio()

	def find_min_crop(self, xi, yi, xa, ya, w, h):
		xc1, yc1 = xi, yi
		xc2 = xa
		wc = xa - xi

		yc2 = (xa - xi) * h / w + yi
		if yc2 < ya:
			yc2 = ya
			xc2 = (ya - yi) * w / h + xi
			if xc2 > w:
				xc2 = w
				xc1 = w - (ya - yi) * w / h
		elif yc2 > h:
			yc2 = h
			yc1 = h - wc * h / w
		return round(xc1), round(yc1), round(xc2), round(yc2)

	def crop_xy_ratio(self):
		xc1_minc, yc1_minc, xc2_minc, yc2_minc = self.find_min_crop(self.xmin_r, self.ymin_r, self.xmax_r,
																	self.ymax_r,
																	self.w_r, self.h_r)
		xc1 = round(xc1_minc * random.random())
		yc1 = round(yc1_minc * random.random())

		xc2 = self.xmax_r + round((self.w_r - self.xmax_r) * random.random())
		yc2 = (xc2 - xc1) * self.h_r / self.w_r + yc1
		if yc2 < self.ymax_r:
			yc2 = self.ymax_r
			xc2 = (yc2 - yc1) * self.w_r / self.h_r + xc1
			if xc2 > self.w_r:
				xc2 = self.w_r
				xc1 = self.w_r - (yc2 - yc1) * self.w_r / self.h_r
		elif yc2 > self.h_r:
			yc2 = self.h_r
			yc1 = self.h_r - (xc2 - xc1) * self.h_r / self.w_r
		wc = xc2 - xc1
		hc = yc2 - yc1
		if xc1 < 0 or yc1 < 0 or xc2 > self.w_r or yc2 > self.h_r:
			print(xc1, yc1, xc2, yc2, 'min:', xc1_minc, yc1_minc, xc2_minc, yc2_minc, 'w, h:', self.w_r, self.h_r)

		self.xmin_rrc = round((self.xmin_r - xc1) / wc * self.args.resize)
		self.xmax_rrc = round((self.xmax_r - xc1) / wc * self.args.resize)
		self.ymin_rrc = round((self.ymin_r - yc1) / hc * self.args.resize)
		self.ymax_rrc = round((self.ymax_r - yc1) / hc * self.args.resize)
		return xc1, yc1, xc2, yc2


