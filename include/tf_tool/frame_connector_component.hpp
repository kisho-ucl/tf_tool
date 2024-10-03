// #ifndef ROS2_RS_PCL__RSPCL_FILTER_COMPONENT_HPP_
// #define ROS2_RS_PCL__RSPCL_FILTER_COMPONENT_HPP_

// rclcpp
#include <rclcpp/rclcpp.hpp>
// message
#include <tf2_msgs/msg/tf_message.hpp>
// others
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

using namespace std;

class FrameConnectorComponent : public rclcpp::Node
{
  public:
    FrameConnectorComponent(const rclcpp::NodeOptions & options);
    // parent_frame to child_frame
    rcl_interfaces::msg::SetParametersResult CB_param_reset(const std::vector<rclcpp::Parameter> & params);
    string frame_p;
    string frame_c;
    double p2c_x;
    double p2c_y;
    double p2c_z;
    double p2c_roll;
    double p2c_pitch;
    double p2c_yaw;
     
  private:
    OnSetParametersCallbackHandle::SharedPtr reset_param_handler_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster_;
    // std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    void set_init_param();
    void set_new_tf();
};

// #endif // ROS2_RS_PCL__RSPCL_FILTER_COMPONENT_HPP_