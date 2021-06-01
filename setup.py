from os import path

import setuptools

# Read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='pipeline',
    version='0.1',
    author="Washington Cunha",
    author_email="washingtoncunha@dcc.ufmg.br",
    description="Extended pre-processing pipeline for text classification: On the role of meta-feature representations, sparsification and selective sampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/waashk/extended-pipeline",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: Ubuntu 18.04",
    ],
)
