import os
import torch
import numpy as np

from net import DiffModel
import torch.nn.functional as F
from something import load_initialize
from PIL import Image
from decom import Decom
from denoise import generalized_steps
from torchvision.utils import save_image


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args):
        self.args = args
        self.device_ids = self.args.device
        if torch.cuda.is_available() and len(self.device_ids) > 0:
            devices = [torch.device('cuda:' + str(device)) for device in self.device_ids]
            torch.cuda.set_device(devices[0])

        self.device = devices[0]
        self.model_var_type = args.model_var_type
        betas = get_beta_schedule(
            beta_schedule=args.beta_schedule,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            num_diffusion_timesteps=args.T,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        self.decom_model = Decom().to(self.device)
        self.decom_model = load_initialize(self.decom_model, self.args.decom_path)

    def sample_img(self, x, cond, model, illu, color):
        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)

            beta = self.betas.to(x.device)
            xs, _ = generalized_steps(x, cond, illu, color, seq, model, beta, eta=0.0)
            x = xs[-1]

        return x


    def test_all(self):
        with torch.no_grad():
            model = DiffModel(self.args).to(self.device)
            model = torch.nn.DataParallel(model)
            # pre_dir = os.path.join(self.args.ckpt_path, self.args.pre_ckpt)
            pre_dir = self.args.pre_ckpt
            load_checkpoint(model, pre_dir, self.device)
            model.eval()

            filePath = self.args.test_folder

            for subfolder in os.listdir(filePath):
                subfolder_path = os.path.join(filePath, subfolder)
                save_folder = os.path.join(self.args.sampled_dir, subfolder)
                os.makedirs(save_folder, exist_ok=True)
                for file_name in os.listdir(subfolder_path):
                    test_name = os.path.join(subfolder_path, file_name)
                    print(test_name)
                    Img = Image.open(test_name).convert('RGB')
                    Img = (np.asarray(Img) / 255.0)
                    Img = torch.from_numpy(Img).float()
                    Img = Img.permute(2, 0, 1)
                    Img = Img.unsqueeze(0).to(self.device)

                    b, c, h, w = Img.shape

                    img = pad_to_multiple_of_16(Img)
                    noise = torch.randn_like(img, device=self.device)

                    refl, illu = self.decom_model(Img)
                    illu = illu ** 0.45

                    refl = data_transform(refl)
                    illu = data_transform(illu)
                    refl = pad_to_multiple_of_16(refl)
                    illu = pad_to_multiple_of_16(illu)

                    img = data_transform(img)

                    result = self.sample_img(noise, cond=img, model=model, illu=illu, color=refl)
                    result = inverse_data_transform(result)
                    result = crop_to_original_size(result, h, w)

                    save_image(result, os.path.join(save_folder, file_name))



def load_checkpoint(model, path, device):

    checkpoint = torch.load(path, map_location=device)

    model.module.down.load_state_dict(checkpoint['state_dict']['model_down'])
    model.module.up.load_state_dict(checkpoint['state_dict']['model_up'])
    model.module.mid.load_state_dict(checkpoint['state_dict']['model_mid'])
    model.module.in_out.load_state_dict(checkpoint['state_dict']['model_in_out'])
    model.module.feature_map.load_state_dict(checkpoint['state_dict']['feature'])
    model.module.time_embed.load_state_dict(checkpoint['state_dict']['time_embed'])
    model.module.transformer.load_state_dict(checkpoint['state_dict']['transformer'])
    model.module.channelattn.load_state_dict(checkpoint['state_dict']['channel'])


def pad_to_multiple_of_16(image):
    b, c, h, w = image.size()

    # 计算需要添加的行数和列数
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    # 使用torch.nn.functional.pad函数进行填充
    padded_image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), 'constant', 0)

    return padded_image


def crop_to_original_size(image, original_h, original_w):
    # 获取填充后图像的形状信息
    b, c, h, w = image.size()

    # 计算需要裁剪的行数和列数
    crop_h = h - original_h
    crop_w = w - original_w

    # 使用切片操作进行裁剪
    cropped_image = image[:, :, 0:h - crop_h, 0:w - crop_w]

    return cropped_image


def data_transform(X):
    X = 2 * X - 1.0
    return X

def inverse_data_transform(X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)


