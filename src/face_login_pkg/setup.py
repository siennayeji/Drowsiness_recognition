from setuptools import find_packages, setup
import os

package_name = 'face_login_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), ['launch/drowsiness_system_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sienna',
    maintainer_email='sienna@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'face_identifier_node = face_login_pkg.face_identifier_node:main',
            'firebase_uploader_node = face_login_pkg.firebase_uploader_node:main',
            'drowsiness_logger_node = face_login_pkg.drowsiness_logger_node:main',
             'drowsiness_detection_node = face_login_pkg.drowsiness_detection_node:main',
             'usb_camera_node = face_login_pkg.usb_camera_node:main',
'face_detection_node = face_login_pkg.face_detection_node:main',
        ],
    },
)
