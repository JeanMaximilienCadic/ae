from ae.models import Autoencoder
from torch.nn import MSELoss
import torch
from ae import format_img
from gnutools.fs import listfiles
from ae import cfg
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
criterion = MSELoss()
model = Autoencoder()
model.load_state_dict(torch.load(cfg.fs.model_path))

files = listfiles("test", [".png"])[:cfg.trainer.nsamples]
dataloader = [format_img(img) for img in files]
dataloader = np.array(dataloader).reshape(-1,
                                          1,
                                          1,
                                          cfg.img.W,
                                          cfg.img.W)
dataloader = torch.Tensor(dataloader)
losses = []
for data, path in tqdm(zip(dataloader, files), total=len(dataloader), desc="Processing"):
    img = data
    img = Variable(img, requires_grad=False)
    # ===================forward=====================
    output = model(img)
    loss = criterion(output, img)
    losses.append(float(loss.data.numpy()))
    print(losses[-1])
    if losses[-1]<0.0086:
        print(path)