#include "tf_tool/l2c_projector_component.hpp"

#include <math.h> /* M_PI */
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

using namespace std::chrono_literals;

L2CProjectorComponent::L2CProjectorComponent(const rclcpp::NodeOptions & options)
: Node("hoge")
{
  // dynamic_reconfigure
  // this->declare_parameter<double>("tf_parent2child.x", 0.0);

  // Initialize the transform listener

  // set camera_info from yaml
  set_init_param();

  // callback
  subscriber_pc = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "input_points", 
    rclcpp::SensorDataQoS(), 
    std::bind(&ColoringComponent::CB_cloud, this, std::placeholders::_1)\
  );

  subscriber_im = this->create_subscription<sensor_msgs::msg::Image>(
    "input_image_raw", 
    rclcpp::SensorDataQoS(), 
    std::bind(&ColoringComponent::CB_image, this, std::placeholders::_1)\
  );

  publisher_projected = this->create_publisher<sensor_msgs::msg::Image>("/projected", 10);
}


void L2CProjectorComponent::set_init_param()
{
  // define camera_info param
  sensor_msgs::msg::CameraInfo cam_info;
  RCLCPP_INFO(this->get_logger(), "define camera_info params");
  declare_parameter("height", 0);
  declare_parameter("width", 0);
  declare_parameter("distortion_model", "distortion_model_hoge");
  declare_parameter("header.frame_id", "frame_id_hoge");
  declare_parameter("d", std::vector<float_t>({0.0, 0.0, 0.0, 0.0, 0.0}));
  declare_parameter("k", std::vector<float_t>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
  declare_parameter("r", std::vector<float_t>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
  declare_parameter("p", std::vector<float_t>({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
  declare_parameter("binning_x", 0);
  declare_parameter("binning_y", 0);
  declare_parameter("roi.x_offset", 0);
  declare_parameter("roi.y_offset", 0);
  declare_parameter("roi.height", 0);
  declare_parameter("roi.width", 0);
  declare_parameter("roi.do_rectify", false);
  
  // set camera_info param
  RCLCPP_INFO(this->get_logger(),"set camera_info params");
  cam_info.height = get_parameter("height").as_int();
  cam_info.width = get_parameter("width").as_int();
  cam_info.distortion_model = get_parameter("distortion_model").as_string();
  cam_info.header.frame_id = get_parameter("header.frame_id").as_string();

  std::vector<double> d_vec;
  get_parameter("d", d_vec);
  cam_info.d = d_vec;
  std::vector<double> k_vec;
  get_parameter("k", k_vec);
  for(int i = 0; i < 9; i++)cam_info.k[i] = k_vec[i];
  std::vector<double> r_vec;
  get_parameter("r", r_vec);
  for(int i = 0; i < 9; i++) cam_info.r[i] = r_vec[i];
  std::vector<double> p_vec;
  get_parameter("p", p_vec);
  for(int i = 0; i < 12; i++) cam_info.p[i] = p_vec[i];
  cam_info.binning_x = get_parameter("binning_x").as_int();
  cam_info.binning_y = get_parameter("binning_y").as_int();
  cam_info.roi.x_offset = get_parameter("roi.x_offset").as_int();
  cam_info.roi.y_offset = get_parameter("roi.y_offset").as_int();
  cam_info.roi.height = get_parameter("roi.height").as_int();
  cam_info.roi.width = get_parameter("roi.width").as_int();
  cam_info.roi.do_rectify = get_parameter("roi.do_rectify").as_bool();
  cam_model.fromCameraInfo(cam_info);
}

void L2CProjectorComponent::CB_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
{
  if(!is_received_image) return;

  // set colored_cloud
  pcl::PointCloud<PointXYZIRGB>::Ptr trans_cloud(new pcl::PointCloud<PointXYZIRGB>); //新しいXYZRGB
  pcl::fromROSMsg(*cloud_msg, *trans_cloud);  //受信した/camera/pointcloud をpclに変換 

  pcl::PointCloud<PointXYZIRGB>::Ptr colored_cloud(new pcl::PointCloud<PointXYZIRGB>); //新しいXYZRGB
  pcl::transformPointCloud (*trans_cloud, *colored_cloud, matrix_c2l);

  // set rgb_image and projection_image
  auto cv_image = cv_bridge::toCvShare(last_image_msg, last_image_msg->encoding);
  cv::Mat rgb_image;
  cv::cvtColor(cv_image->image ,rgb_image, CV_BGR2RGB);
  
  // copy or black 
  cv::Mat projection_image = rgb_image.clone();

  for(typename pcl::PointCloud<PointXYZIRGB>::iterator pt=colored_cloud->points.begin(); pt<colored_cloud->points.end(); pt++)
  {   
    if((*pt).z < 0) continue;

      cv::Point3d pt_cv((*pt).x, (*pt).y, (*pt).z);
      cv::Point2d uv;
      uv = cam_model.project3dToPixel(pt_cv);
      // RCLCPP_INFO(this->get_logger(), "uv.x:%f, uv.y:$%f", uv.x, uv.y);
      if(uv.x>0 && uv.x < rgb_image.cols && uv.y > 0 && uv.y < rgb_image.rows)
      {
          // Coloring PointCloud
          (*pt).b = rgb_image.at<cv::Vec3b>(uv)[0];
          (*pt).g = rgb_image.at<cv::Vec3b>(uv)[1];
          (*pt).r = rgb_image.at<cv::Vec3b>(uv)[2];
          // Projection PointCloud
          double range = sqrt( pow((*pt).x, 2.0) + pow((*pt).y, 2.0) + pow((*pt).z, 2.0));
          COLOUR c = GetColour(int(range/20*255.0), 0, 255);
          cv::circle(projection_image, uv, 1, cv::Scalar(int(255*c.b),int(255*c.g),int(255*c.r)), -1);
      }
  }

  pcl::PointCloud<PointXYZIRGB>::Ptr output_cloud(new pcl::PointCloud<PointXYZIRGB>); //新しいXYZRGB
  pcl::transformPointCloud (*colored_cloud, *output_cloud, matrix_l2c);

  sensor_msgs::msg::PointCloud2 sensor_msg;
  pcl::toROSMsg(*output_cloud, sensor_msg);
  publisher_pc->publish(sensor_msg);

  // if(is_received_image){
  //   auto cv_img = cv_bridge::toCvShare(last_image_msg, last_image_msg->encoding);
  //   cv::cvtColor(cv_img->image, targeted_image_, cv::COLOR_BGR2RGB);
  //   RCLCPP_INFO(this->get_logger(), "col:%d, row:$%d", targeted_image_.cols, targeted_image_.rows);
  // }
}

void L2CProjectorComponent::CB_image(const sensor_msgs::msg::Image::SharedPtr image_msg)
{
  // RCLCPP_INFO(this->get_logger(), "get image");
  last_image_msg = image_msg;
  is_received_image = true;
}