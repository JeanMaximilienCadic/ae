import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from ae.models import Autoencoder
from ae import to_img
import os


class Trainer:
    def __init__(self,
                 model : Autoencoder,
                 epochs,
                 continue_from,
                 optimizer,
                 kwargs_optim,
                 criterion,
                 ):
        self._model = model
        self._continue_from = continue_from
        self._epochs=epochs
        self._criterion = criterion()
        self._optimizer = optimizer(model.parameters(), **kwargs_optim)
        self._model.load_state_dict(torch.load(continue_from)) if os.path.exists(continue_from) else None
        self._bar = tqdm(range(self._epochs), desc=">> Training")

    def save(self, pth):
        torch.save(self._model.state_dict(), pth)

    def __call__(self, dataloader, *args, **kwargs):
        last_mu = None
        for epoch in self._bar:
            avg_loss = []
            for data in dataloader:
                img = data
                img = Variable(img, requires_grad=False)
                # ===================forward=====================
                output = self._model(img)
                loss = self._criterion(output, img)
                # ===================backward====================
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                # ===================log========================
                avg_loss.append(loss.data)
            mu = np.mean(avg_loss)
            last_mu = mu if last_mu is None else last_mu
            desc, asc = False, False
            if last_mu<mu:
                asc=True
            elif last_mu>mu:
                desc=True
            self._bar.set_description(f">> Training | loss \t{'⤵' if desc else ('⤴' if asc else '⤳' ) }\t:\t{mu :.4f}")
            if epoch % 100 == 0:
                if desc:
                    self.save(self._continue_from)
                pic = to_img(output.cpu().data)
                save_image(pic, './dc_img/image_{}.png'.format(epoch))
