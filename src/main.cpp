#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tf/transform_broadcaster.h>

#include "AprilTags/TagDetector.h"
#include "AprilTags/Tag16h5.h"
#include "AprilTags/Tag25h7.h"
#include "AprilTags/Tag25h9.h"
#include "AprilTags/Tag36h9.h"
#include "AprilTags/Tag36h11.h"

#define TAG_INDEX 0

#include <cmath>

#ifndef PI
const double PI = 3.14159265358979323846;
#endif
const double TWOPI = 2.0*PI;

/**
 * Normalize angle to be within the interval [-pi,pi].
 */
inline double standardRad(double t) {
  if (t >= 0.) {
    t = fmod(t+PI, TWOPI) - PI;
  } else {
    t = fmod(t-PI, -TWOPI) + PI;
  }
  return t;
}

void wRo_to_euler(const Eigen::Matrix3d& wRo, double& yaw, double& pitch, double& roll) {
    yaw = standardRad(atan2(wRo(1,0), wRo(0,0)));
    double c = cos(yaw);
    double s = sin(yaw);
    pitch = standardRad(atan2(-wRo(2,0), wRo(0,0)*c + wRo(1,0)*s));
    roll  = standardRad(atan2(wRo(0,2)*s - wRo(1,2)*c, -wRo(0,1)*s + wRo(1,1)*c));
  }

static const std::string OPENCV_WINDOW = "Image window";

class ImageConverter {
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;



  AprilTags::TagDetector* m_tagDetector;
  AprilTags::TagCodes m_tagCodes;

  bool m_draw; // draw image and April tag detections?
 
  int m_width; // image size in pixels
  int m_height;
  double m_tagSize; // April tag side length in meters of square black frame
  double m_fx; // camera focal length in pixels
  double m_fy;
  double m_px; // camera principal point
  double m_py;

public:
  ImageConverter()
    : it_(nh_),
    m_tagDetector(NULL),
    m_tagCodes(AprilTags::tagCodes16h5),

    m_draw(true),
 
    m_width(640),
    m_height(480),
    m_tagSize(0.166),
    m_fx(600),
    m_fy(600),
    m_px(m_width/2),
    m_py(m_height/2)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/image_raw", 1, &ImageConverter::DetectTag, this);
    image_pub_ = it_.advertise("/image_converter/output_video", 1);

    cv::namedWindow(OPENCV_WINDOW);

    m_tagDetector = new AprilTags::TagDetector(m_tagCodes);
  }

  ~ImageConverter() {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void print_detection(AprilTags::TagDetection& detection) const {
    cout << "  Id: " << detection.id
         << " (Hamming: " << detection.hammingDistance << ")";

    // recovering the relative pose of a tag:

    // NOTE: for this to be accurate, it is necessary to use the
    // actual camera parameters here as well as the actual tag size
    // (m_fx, m_fy, m_px, m_py, m_tagSize)

    Eigen::Vector3d translation;
    Eigen::Matrix3d rotation;
    detection.getRelativeTranslationRotation(m_tagSize, m_fx, m_fy, m_px, m_py,
                                             translation, rotation);

    Eigen::Matrix3d F;
    F <<
      1, 0,  0,
      0,  -1,  0,
      0,  0,  1;
    Eigen::Matrix3d fixed_rot = F*rotation;
    double yaw, pitch, roll;
    wRo_to_euler(fixed_rot, yaw, pitch, roll);

    cout << "  distance=" << translation.norm()
         << "m, x=" << translation(0)
         << ", y=" << translation(1)
         << ", z=" << translation(2)
         << ", yaw=" << yaw
         << ", pitch=" << pitch
         << ", roll=" << roll
         << endl;

    // Also note that for SLAM/multi-view application it is better to
    // use reprojection error of corner points, because the noise in
    // this relative pose is very non-Gaussian; see iSAM source code
    // for suitable factors.
  }

  void DetectTag(const sensor_msgs::ImageConstPtr& msg) {
  static tf::TransformBroadcaster br_;
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } 
    catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat image = cv_ptr->image;
    cv::Mat image_gray;
    cv::cvtColor(image, image_gray, CV_BGR2GRAY);
    vector<AprilTags::TagDetection> detections = m_tagDetector->extractTags(image_gray);

    cout << detections.size() << " tags detected:" << endl;
//    for (int i = 0; i < detections.size(); i++) {
//      print_detection(detections[i]);
//    }


    for (int i = 0; i < int(detections.size()); i++) {
      if(detections[i].id == TAG_INDEX) {
        Eigen::Matrix4d H = detections[i].getRelativeTransform(m_tagSize, m_fx, m_fy, m_px, m_py);
	Eigen::Matrix3d R = H.block<3, 3>(0, 0);
        Eigen::Vector3d T = H.block<3, 1>(0, 3);

	Eigen::Vector3d origin = -R.transpose() * T;
	Eigen::Matrix3d Rt = R.transpose();
	tf::Matrix3x3 basis;
	basis.setValue(Rt(0, 0), Rt(0, 1), Rt(0, 2), 
                       Rt(1, 0), Rt(1, 1), Rt(1, 2), 
                       Rt(2, 0), Rt(2, 1), Rt(2, 2));
        // Publish tf
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(origin(0), origin(1), origin(2)));
        transform.setBasis(basis);
        br_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "tag0"));
      }
    }

    for (int i = 0; i < int(detections.size()); i++) {
          // also highlight in the image
      detections[i].draw(image);
    }
    cv::imshow(OPENCV_WINDOW, image);
    cv::waitKey(3);


    // Output modified video stream
//    image_pub_.publish(cv_ptr->toImageMsg());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "tag_detection");
  ImageConverter ic;
  ros::spin();
  return 0;
}
