import os
from glob import glob

from setuptools import setup

package_name = 'tf_tool'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        (os.path.join('share', package_name), glob('launch/*launch.py')),
    ],
)