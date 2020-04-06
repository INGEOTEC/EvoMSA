# Copyright 2017 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup
import EvoMSA
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension
import sys
version = EvoMSA.__version__

extension = [Extension('EvoMSA.cython_utils', ["EvoMSA/cython_utils.pyx"],
                       include_dirs=[np.get_include()])]

with open('README.rst') as fpt:
    long_desc = fpt.read()

setup(
    name="EvoMSA",
    description="""Sentiment Analysis System based on B4MSA and EvoDAG""",
    long_description=long_desc,
    version=version,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        'Programming Language :: Python :: 3',
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
    url='https://github.com/ingeotec/EvoMSA',
    author="Mario Graff",
    author_email="mgraffg@ieee.org",
    packages=['EvoMSA', 'EvoMSA/tests'],
    include_package_data=True,
    zip_safe=False,
    ext_modules=cythonize(extension,
                          compiler_directives={'language_level': sys.version_info[0],
                                               'profile': False,
                                               'nonecheck': False,
                                               'boundscheck': False}),
    package_data={'EvoMSA/tests': ['tweets.json'],
                  '': ['*.pxd']},
    install_requires=['B4MSA', 'ConceptModelling'],
    entry_points={
        'console_scripts': ['EvoMSA-train=EvoMSA.command_line:train',
                            'EvoMSA-predict=EvoMSA.command_line:predict',
                            'EvoMSA-performance=EvoMSA.command_line:performance']
    }
)
