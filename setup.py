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
