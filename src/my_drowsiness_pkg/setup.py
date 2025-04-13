from setuptools import setup

package_name = 'my_drowsiness_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/drowsiness_system.launch.py']),
        ('share/' + package_name + '/weights', ['my_drowsiness_pkg/weights/cnn_lstm_drowsiness_1.pth']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sienna',
    maintainer_email='you@example.com',
    description='ROS 2 drowsiness detection in pure Python',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_input_node = my_drowsiness_pkg.camera_input_node:main',
            'sequence_buffer_node = my_drowsiness_pkg.sequence_buffer_node:main',
            'drowsiness_node = my_drowsiness_pkg.drowsiness_node:main',
            'alert_node = my_drowsiness_pkg.alert_node:main',
            'drowsiness_visual_node = my_drowsiness_pkg.drowsiness_visual_node:main',
        ],
    },
)
