import torch
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from gnutools.fs import listfiles
import cv2
from tqdm import tqdm


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, W, W)
    return x


def format_img(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (W, W))
    img =img/255
    img*=2
    img-=1
    return img


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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
                pic = to_img(output.cpu().data)
                save_image(pic, './dc_img/image_{}.png'.format(epoch))


if __name__ == "__main__":
    W=400
    bs = 16
    model_path = 'pth/conv_autoencoder.pth'
    #Dataloader
    dataloader = [format_img(img) for img in listfiles("data", [".png"])[:80]]
    dataloader = np.array(dataloader).reshape(-1, bs, 1, W, W)
    dataloader = torch.Tensor(dataloader)
    trainer = Trainer(model=Autoencoder(),
                      epochs=100,
                      criterion=nn.MSELoss,
                      optimizer= torch.optim.Adam,
                      kwargs_optim={
                          "lr": 1e-3,
                          "weight_decay": 1e-5
                      },
                      continue_from=model_path
                      )
    trainer(dataloader)
    trainer.save(model_path)
