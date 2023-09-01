# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools, platform

with open("README.md", "r", encoding='utf_8') as fh:
    long_description = fh.read()

install_requires=[
    'einops'
]

setuptools.setup(
    name="gptplay",
    version="0.1.0",
    author="Shital Shah",
    description="Playground for autoregresive models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sytelus/grokking",
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
