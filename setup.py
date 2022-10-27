import os
import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as f:
    long_description = f.read()

# read the contents of requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name = 'MedCLIP',
    version = '0.0.3',
    author = 'Zifeng Wang',
    author_email = 'zifengw2@illinois.edu',
    description = 'Contrastive Learning from Medical Images and Text.',
    url = 'https://github.com/RyanWangZf/MedCLIP',
    keywords=['vision-language model','X-ray','deep learning','AI','healthcare'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)