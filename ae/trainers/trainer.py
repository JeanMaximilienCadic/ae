import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from ae.models import Autoencoder

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
        self._epochs=epochs
        self._criterion = criterion()
        self._optimizer = optimizer(model.parameters(), **kwargs_optim)
        self._model.load_state_dict(torch.load(continue_from))
        self._bar = tqdm(range(self._epochs), desc=">> Training")

    def save(self, pth):
        torch.save(self._model.state_dict(), pth)

    def __call__(self, dataloader, *args, **kwargs):
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
            self._bar.set_description(f'>> Training | loss:{np.mean(avg_loss) :.4f}')
            if epoch % 10 == 0:
                pic = self.to_img(output.cpu().data)
                save_image(pic, './dc_img/image_{}.png'.format(epoch))
