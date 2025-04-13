from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_drowsiness_pkg',
            executable='camera_input_node',
            name='camera_input_node',
            output='screen'
        ),
        Node(
            package='my_drowsiness_pkg',
            executable='sequence_buffer_node',
            name='sequence_buffer_node',
            output='screen'
        ),
        Node(
            package='my_drowsiness_pkg',
            executable='drowsiness_node',
            name='drowsiness_node',
            output='screen'
        ),
        Node(
            package='my_drowsiness_pkg',
            executable='alert_node',
            name='alert_node',
            output='screen'
        )
    ])
