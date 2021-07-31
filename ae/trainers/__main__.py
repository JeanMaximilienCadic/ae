from ae import cfg, format_img
import numpy as np
from gnutools.fs import listfiles
from ae.trainers import Trainer
import torch
from ae.models import Autoencoder
from torch import nn


dataloader = [format_img(img) for img in listfiles(cfg.fs.data, [".png"])[:cfg.trainer.nsamples]]
dataloader = np.array(dataloader).reshape(-1,
                                          cfg.trainer.bs,
                                          1,
                                          cfg.img.W,
                                          cfg.img.W)
dataloader = torch.Tensor(dataloader)

trainer = Trainer(model=Autoencoder(),
                  epochs=cfg.trainer.epochs,
                  criterion=nn.MSELoss,
                  optimizer= torch.optim.Adam,
                  kwargs_optim={
                      "lr": cfg.trainer.lr,
                      "weight_decay": cfg.trainer.wdc
                  },
                  continue_from=cfg.fs.model_path
                  )
trainer(dataloader)
trainer.save(cfg.fs.model_path)
