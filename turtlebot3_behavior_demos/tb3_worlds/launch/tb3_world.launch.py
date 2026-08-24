import os
from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    tb3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    x_pose = LaunchConfiguration("x_pose", default="0.0")
    y_pose = LaunchConfiguration("y_pose", default="0.0")
    yaw_pose = LaunchConfiguration("yaw_pose", default="0.0")

    tb3_world_dir = get_package_share_directory("tb3_worlds")
    default_world = join(tb3_world_dir, "worlds", "extinguisher_room.world")
    world = LaunchConfiguration("world", default=default_world)

    # Start Gazebo server and client
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(gazebo_ros_dir, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world, "pause": "false"}.items(),
    )
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(gazebo_ros_dir, "launch", "gzclient.launch.py")
        )
    )

    # Start robot state publisher
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(tb3_gazebo_dir, "launch", "robot_state_publisher.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # Gazebo can take over 30 seconds to expose its factory service under WSLg.
    # The upstream TurtleBot launch uses spawn_entity.py's 30-second default,
    # so invoke the same spawner explicitly with a startup-safe timeout.
    turtlebot_model = os.environ.get("TURTLEBOT3_MODEL", "waffle_pi")
    turtlebot_sdf = join(
        tb3_gazebo_dir,
        "models",
        f"turtlebot3_{turtlebot_model}",
        "model.sdf",
    )
    spawn_turtlebot_cmd = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-entity", turtlebot_model,
            "-file", turtlebot_sdf,
            "-x", x_pose,
            "-y", y_pose,
            "-z", "0.01",
            "-Y", yaw_pose,
            "-timeout", "180.0",
            "-unpause",
        ],
    )

    return LaunchDescription(
        [
            gzserver_cmd,
            gzclient_cmd,
            robot_state_publisher_cmd,
            spawn_turtlebot_cmd,
        ]
    )
