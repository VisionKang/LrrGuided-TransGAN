import torch
import torch.nn as nn
from net import Base_MODEL, Detail_MODEL, Fusion_MODEL

class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self):
		super(Generator, self).__init__()
		self.base_model = Base_MODEL(img_size=224, patch_size=1, in_chans=3, out_chans=64,
                 embed_dim=96, depths=[4, 2, 2], num_heads=[2, 4, 8],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False)
		self.detail_model = Detail_MODEL(in_channel=3, out_channel=32, kernal_size=3)
		self.fusion_model = Fusion_MODEL(in_channel=64, output_channel=1)

	def forward(self, ir_base, ir_detail, vis_base, vis_detail):
		base_avg = (ir_base + vis_base) / 2
		detail_max = torch.max(ir_detail, vis_detail)
		base_data = torch.cat([ir_base, vis_base, base_avg], 1)
		detail_data = torch.cat([ir_detail, vis_detail, detail_max], 1)
		base = self.base_model(base_data)
		detail = self.detail_model(detail_data)
		fuse_out = self.fusion_model(base, detail)
		return fuse_out

class Discriminator_ir(nn.Module):
	"""docstring for Discriminator_v"""
	def __init__(self):
		super(Discriminator_ir, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 16, 3, 2),
			nn.LeakyReLU(0.2))

		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 32, 3, 2),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2))

		self.conv3 = nn.Sequential(
			nn.Conv2d(32, 64, 3, 2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2))

		self.conv4 = nn.Sequential(
			nn.Conv2d(64, 128, 3, 2),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2))

		self.conv5 = nn.Sequential(
			nn.Conv2d(128, 256, 3, 2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),
			nn.Flatten())

		self.fc = nn.Sequential(
			nn.Linear(256 * 6 * 6, 1),
			nn.Tanh())

	def forward(self, v):
		v = self.conv1(v)
		v = self.conv2(v)
		v = self.conv3(v)
		v = self.conv4(v)
		v = self.conv5(v)
		v = self.fc(v)
		v = v / 2 + 0.5
		return v

class Discriminator_vis(nn.Module):
	"""docstring for Discriminator_i"""
	def __init__(self):
		super(Discriminator_vis, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 16, 3, 2),
			nn.LeakyReLU(0.2))

		self.conv2 = nn.Sequential(
			nn.Conv2d(16, 32, 3, 2),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2))

		self.conv3 = nn.Sequential(
			nn.Conv2d(32, 64, 3, 2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2))

		self.conv4 = nn.Sequential(
			nn.Conv2d(64, 128, 3, 2),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2))

		self.conv5 = nn.Sequential(
			nn.Conv2d(128, 256, 3, 2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2),
			nn.Flatten())

		self.fc = nn.Sequential(
			nn.Linear(256 * 6 * 6, 1),
			nn.Tanh())

	def forward(self, i):
		i = self.conv1(i)
		i = self.conv2(i)
		i = self.conv3(i)
		i = self.conv4(i)
		i = self.conv5(i)
		i = self.fc(i)
		i = i / 2 + 0.5
		return i

class DDcGAN(nn.Module):
	"""docstring for DDcGAN"""
	def __init__(self, if_train=False):
		super(DDcGAN, self).__init__()
		self.if_train = if_train

		self.G = Generator()
		self.D_ir = Discriminator_ir()
		self.D_vis = Discriminator_vis()


	def forward(self, ir_base, ir_detail, vis_base, vis_detail, ir, vis):
		fusion = self.G(ir_base, ir_detail, vis_base, vis_detail)
		if self.if_train:
			score_pan = self.D_vis(vis)
			score_ms = self.D_ir(ir)
			score_g_pan = self.D_vis(fusion)
			score_g_ms = self.D_ir(fusion)
			return score_pan, score_ms, score_g_pan, score_g_ms, fusion
		else:
			return fusion

if __name__=='__main__':
	vis=torch.rand((1,1,256,256))
	ir=torch.rand((1,1,64,64))
	model=Discriminator_ir()
	output=model(ir)
	print(output.shape)