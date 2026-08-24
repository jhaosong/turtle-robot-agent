from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from os.path import join


def generate_launch_description():
    tb3_nav2_dir = get_package_share_directory("turtlebot3_navigation2")
    tb3_world_dir = get_package_share_directory("tb3_worlds")
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    # Spawn the world and robot
    spawn_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(tb3_world_dir, "launch", "tb3_world.launch.py")
        ),
        launch_arguments={
            "x_pose": LaunchConfiguration("x_pose", default=-3.5),
            "y_pose": LaunchConfiguration("y_pose", default=-3.5),
            "yaw_pose": LaunchConfiguration("yaw_pose", default=0.0),
            "world": LaunchConfiguration(
                "world",
                default=join(tb3_world_dir, "worlds", "extinguisher_room.world"),
            ),
        }.items(),
    )
    # Let Gazebo initialize before the navigation stack begins discovery.
    spawn_world_delayed = TimerAction(period=3.0, actions=[spawn_world])

    # Start navigation stack
    default_map = join(tb3_world_dir, "maps", "extinguisher_room_map.yaml")
    nav_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(tb3_nav2_dir, "launch", "navigation2.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "map": LaunchConfiguration("map", default=default_map),
        }.items(),
    )

    # Set AMCL initial pose
    amcl_init_pose = Node(
        package="tb3_worlds",
        executable="set_init_amcl_pose.py",
        name="init_pose_publisher",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "x": LaunchConfiguration("x_pose", default=-3.5),
                "y": LaunchConfiguration("y_pose", default=-3.5),
                "theta": LaunchConfiguration("yaw_pose", default=0.0),
            }
        ],
    )

    # Spawn the semantic-search target in front of its named observation pose.
    spawn_fire_extinguisher = Node(
        package="tb3_worlds",
        executable="fire_extinguisher_spawner.py",
        name="fire_extinguisher_spawner",
        parameters=[{
            "x": LaunchConfiguration("object_x", default=0.0),
            "y": LaunchConfiguration("object_y", default=0.0),
            "z": LaunchConfiguration("object_z", default=0.30),
            "yaw": LaunchConfiguration("object_yaw", default=0.0),
        }],
    )

    return LaunchDescription(
        [spawn_world_delayed, nav_stack, amcl_init_pose, spawn_fire_extinguisher]
    )
