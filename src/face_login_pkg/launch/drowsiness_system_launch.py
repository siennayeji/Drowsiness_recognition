from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='face_login_pkg',
            executable='usb_camera_node',
            name='usb_camera_node',
            output='screen'
        ),
        Node(
            package='face_login_pkg',
            executable='face_detection_node',
            name='face_detection_node',
            output='screen'
        ),
        Node(
            package='face_login_pkg',
            executable='drowsiness_detection_node',
            name='drowsiness_detection_node',
            output='screen'
        ),
        Node(
            package='face_login_pkg',
            executable='drowsiness_logger_node',
            name='drowsiness_logger_node',
            output='screen'
        )
    ])
