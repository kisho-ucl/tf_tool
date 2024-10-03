#include "rclcpp/rclcpp.hpp"
#include "tf_tool/frame_connector_component.hpp"

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);

    auto frame_connector = std::make_shared<FrameConnectorComponent>(options);
    rclcpp::spin(frame_connector);
    
    rclcpp::shutdown();
    return 0;
}