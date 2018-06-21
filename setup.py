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
version = EvoMSA.__version__

setup(
    name="EvoMSA",
    description="""Sentiment Analysis System based on B4MSA and EvoDAG""",
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
    package_data={'EvoMSA/conf': ['default_parameters.json'],
                  'EvoMSA/tests': ['tweets.json']},
    install_requires=['B4MSA', 'EvoDAG'],
    entry_points={
        'console_scripts': ['EvoMSA-train=EvoMSA.command_line:train',
                            'EvoMSA-predict=EvoMSA.command_line:predict',
                            'EvoMSA-utils=EvoMSA.command_line:utils',
                            'EvoMSA-performance=EvoMSA.command_line:performance']
    }
)
