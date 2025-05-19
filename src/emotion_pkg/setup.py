from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'emotion_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'model'), glob('emotion_pkg/model/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sienna',
    maintainer_email='siennayeji@konkuk.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'emotion_publish_node = emotion_pkg.emotion_publish_node:main',
            'emotion_reaction_node = emotion_pkg.emotion_reaction_node:main',
        ],
    },
)
