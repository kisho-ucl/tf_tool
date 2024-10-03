import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, Bool, Float32, Int32, Header
import open3d as o3d
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial.transform import Rotation
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf2_ros
import geometry_msgs

n=4
pcd_list = []
T0_list = []
T1_list = []
T2_list = []
T3_list = []

dir_origin  = f'/home/realsense/kisho_ws/ros_ws/src/tf_tool/src/frame_connector/data_pokemon_nonmoving2/data0'
T2_origin = np.load(f'{dir_origin}/trans2.npy')

for i in range(n):
    dir0 = f'/home/realsense/kisho_ws/ros_ws/src/tf_tool/src/frame_connector/data_pokemon_nonmoving2/data{i}'
    #dir0 = f'/home/realsense/kisho_ws/ubicomp/3d_reconstruction/src/yolo_tools/yolo_tools/dataset_paramEst/data_pokemon_moving/data{i}'
    pcd = o3d.io.read_point_cloud(f"{dir0}/new_pcd.ply")
    T0 = np.load(f"{dir0}/trans0.npy")
    T1 = np.load(f"{dir0}/trans1.npy")
    T2 = np.load(f"{dir0}/trans2.npy")
    T3 = np.load(f"{dir0}/trans3.npy")
    T0_list.append(T0)
    T1_list.append(T1)
    T2_list.append(T2)
    T3_list.append(T3)
    pcd_list.append(pcd)



#pcd0.transform(T0)
#pcd2.transform(T2)
#o3d.visualization.draw_geometries([pcd0,pcd2])


def copy_pcd(pcd):
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    pcd2.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    return pcd2

class TfListenerNode(Node):
    def __init__(self):
        super().__init__('tf_listener_node')

        # Create a tf2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Call the function periodically to get the transform
        self.timer = self.create_timer(0.5, self.get_tf)
        self.pub_pcd = self.create_publisher(PointCloud2, '/pointcloud', 10)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.publish_static_tf()

    def publish_static_tf(self):
        static_tfs = []
        t1 = geometry_msgs.msg.TransformStamped()
        t1.header.stamp = self.get_clock().now().to_msg()
        t1.header.frame_id = 'map'
        t1.child_frame_id = 'robot_base'

        t1.transform.translation.x = 0.0
        t1.transform.translation.y = 0.0
        t1.transform.translation.z = 0.0
        
        t1.transform.rotation.x = 0.0
        t1.transform.rotation.y = 0.0
        t1.transform.rotation.z = 0.0
        t1.transform.rotation.w = 1.0
        static_tfs.append(t1)
        #self.static_broadcaster.sendTransform(t1)

        
        # end_effectorからcamera_linkへのTransform
        t2 = geometry_msgs.msg.TransformStamped()
        t2.header.stamp = self.get_clock().now().to_msg()
        t2.header.frame_id = 'robot_base'
        t2.child_frame_id = 'end_effector'
        

        t2.transform.translation.x = 0.0
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.1

        t2.transform.rotation.x = 0.0
        t2.transform.rotation.y = 0.0
        t2.transform.rotation.z = 0.0
        t2.transform.rotation.w = 1.0
        static_tfs.append(t2)

        self.static_broadcaster.sendTransform(static_tfs)
        self.get_logger().info("Static TF published ")
    
    def get_tf(self):
        try:
            # Lookup transform between end_effector and camera_link
            trans = self.tf_buffer.lookup_transform('end_effector', 'camera_link', rclpy.time.Time())
            self.print_transform(trans)
        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {str(e)}")

    def print_transform(self, trans: TransformStamped):
        self.get_logger().info(f"Transform from {trans.header.frame_id} to {trans.child_frame_id}:")
        self.get_logger().info(f"Translation: {trans.transform.translation.x}, {trans.transform.translation.y}, {trans.transform.translation.z}")
        self.get_logger().info(f"Rotation: {trans.transform.rotation.x}, {trans.transform.rotation.y}, {trans.transform.rotation.z}, {trans.transform.rotation.w}")
        tx,ty,tz = trans.transform.translation.x,trans.transform.translation.y,trans.transform.translation.z
        rx,ry,rz,rw = trans.transform.rotation.x,trans.transform.rotation.y ,trans.transform.rotation.z,trans.transform.rotation.w

        T2_new = np.identity(4)
        rot = Rotation.from_quat(np.array([rx,ry,rz,rw]))
        T2_new[:3,:3] = rot.as_matrix()
        T2_new[:3,3] = np.array([tx,ty,tz])

        pcd = o3d.geometry.PointCloud()
        for i in [0,2]:
            new_pcd = copy_pcd(pcd_list[i])
            T0 = T0_list[i]
            T1 = T1_list[i]
            T2 = T2_new
            #T2 = T2_list[i]
            #T2 = T2_origin
            T3 = T3_list[i]
            T = T0@T1@T2@T3
            new_pcd.transform(T)
            pcd += new_pcd

        pcd_down = pcd.voxel_down_sample(0.001)
        self.publish_pcd_new(pcd_down)


    def publish_pcd_new(self,pcd):
        # Open3DのPointCloudからデータを取得
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # PointCloud2メッセージの作成
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'  # 適切なフレームIDに変更

        # フィールドを定義
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='g', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='b', offset=20, datatype=PointField.FLOAT32, count=1),
        ]

        # データを結合
        points_colors = np.hstack((points, colors.astype(np.float32)))  # カラーをFLOAT32型に変換
        cloud_msg = pc2.create_cloud(header, fields, points_colors)
        self.get_logger().info(f"published pcd")
        self.pub_pcd.publish(cloud_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TfListenerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
