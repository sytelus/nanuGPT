# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools, platform

with open("README.md", "r", encoding='utf_8') as fh:
    long_description = fh.read()

install_requires=[
    'einops', 'tiktoken',
    'wandb', 'mlflow',
    'sentencepiece', 'tokenizers', 'transformers', 'datasets',
    'tqdm', 'matplotlib', 'rich'
]

setuptools.setup(
    name="nanugpt",
    version="0.2.8",
    author="Shital Shah",
    description="Simple, reliable and well tested training code for quick experiments with transformer based models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sytelus/nanugpt",
    #packages=setuptools.find_packages(),
    packages=setuptools.find_packages(),
	license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research'
    ],
    include_package_data=True,
    install_requires=install_requires
)

