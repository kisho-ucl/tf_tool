#include "rclcpp/rclcpp.hpp"
#include "tf_tool/l2c_projector_component.hpp"

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);

    auto l2c_projector = std::make_shared<L2CProjectorComponent>(options);
    rclcpp::spin(l2c_projector);
    
    rclcpp::shutdown();
    return 0;
}