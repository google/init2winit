"""init2winit.

See more details in the
[`README.md`](https://github.com/google/init2winit).
"""

import os
import sys

from setuptools import find_packages
from setuptools import setup

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'init2winit')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

setup(
    name='init2winit',
    version=__version__,
    description='init2winit',
    author='init2winit Team',
    author_email='znado@google.com',
    url='http://github.com/google/init2winit',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        'absl-py>=0.8.1',
        'flax',
        'jax',
        'numpy>=1.7',
        'tensorboard',
        'tensorflow-datasets',
        'tf-nightly',
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
