
<h1 align="center">
  <br>
  <a href="https://drive.google.com/uc?id=1WyX0YZ1raHjmCQry2IZKsTG2Nebl1ch6"><img src="https://drive.google.com/uc?id=1WyX0YZ1raHjmCQry2IZKsTG2Nebl1ch6" alt="AE" width="200"></a>
  <br>
  AE
  <br>
</h1>

<p align="center">
  <a href="#code-structure">Code</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#docker">Docker</a> •
  <a href="#PythonEnv">PythonEnv</a> •

[comment]: <> (  <a href="#notebook">Notebook </a> •)
</p>

Configuration file:
```yaml
project: ae

fs:
  data: "data"
  model_path: 'pth/conv_autoencoder.pth'

img:
  W: 400

trainer:
  bs: 16
  nsamples: 80

```

Main:
```python
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
```

### Code structure
```python
from setuptools import setup
from ae import __version__

setup(
    name="ae",
    version=__version__,
    short_description="ae",
    long_description="ae",
    packages=[
        "ae",
        "ae.trainers",
        "ae.models",
    ],
    include_package_data=True,
    package_data={'': ['*.yml']},
    url='https://github.com/JeanMaximilienCadic/ae',
    license='CMJ',
    author='CADIC Jean-Maximilien',
    python_requires='>=3.8',
    install_requires=[r.rsplit()[0] for r in open("requirements.txt")],
    author_email='me@cadic.jp',
    description='ae',
    platforms="linux_debian_10_x86_64",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ]
)
```

### How to use
To clone and run this application, you'll need [Git](https://git-scm.com) and [ https://docs.docker.com/docker-for-mac/install/]( https://docs.docker.com/docker-for-mac/install/) and Python installed on your computer. 
From your command line:

Install the ae:
```bash
# Clone this repository and install the code
git clone https://github.com/JeanMaximilienCadic/ae

# Go into the repository
cd ae
```


### Docker
```shell
docker build . -t cadic/ae -f docker/Dockerfile
docker run --rm --name cadc_ae -it cadic/ae
```

### PythonEnv
```
python setup.py install
python -m ae
```