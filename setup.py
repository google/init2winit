"""init2winit.

See more details in the
[`README.md`](https://github.com/google/init2winit).
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name='init2winit',
    version='0.0.1',
    description='init2winit',
    author='init2winit Team',
    author_email='znado@google.com',
    url='http://github.com/google/init2winit',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        'absl-py>=0.8.1',
        'clu',
        'flax',
        'jax',
        'jax-bitempered-loss',
        'jraph',
        'ml_collections',
        'numpy>=1.7',
        'optax',
        'optax-shampoo',
        'pandas',
        'sentencepiece',
        'tensorboard',
        'tensorflow-datasets',
        'tensorflow-text==2.5.0-rc0',
        'tensorflow==2.12.1',
    ],
    extras_require={},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='jax machine learning',
)
