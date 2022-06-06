from setuptools import setup

with open('requirements.txt', 'r') as f:
    requirements = [l.strip() for l in f.readlines()]

setup(
    name='odin',
    version='0.1.0',
    description='Odin is an open source diagnosis framework for generic machine learning classification tasks and for computer vision object detection and instance segmentation tasks that lets developers add meta-annotations to their data sets, compute performance metrics split by meta-annotation values, and visualize diagnosis reports. Odin is agnostic to the training platform and input formats and can be extended with application- and domain-specific meta-annotations and metrics with almost no coding.',
    url='',
    author='',
    author_email='',
    license='Apache License 2.0',
    packages=['odin'],
    install_requires=requirements,
)

