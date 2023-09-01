#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H
#include <list>
#include <vector>
#include <Eigen/Core>

#include <ceres/ceres.h>
#include "projection_factor.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "pointcloud_rgbd.hpp"
#include "so3_math.h"
#include "feature_tracker.h"
#include "../optical_flow/lkpyramid.hpp"

using namespace std;
using namespace Eigen;

/**
* @class FeaturePerFrame
* @brief 特征类
* detailed 
*/
class FeaturePerFrame
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //_point:[x,y,z,u,v,vx,vy]
    FeaturePerFrame(const imageFeature &_point)
    {
        point = _point.point_cam;
        uv = _point.point_img;
        velocity = _point.point_vel;
        track_cnt = _point.track_time; 
        // cur_td = td;
    }
    // double cur_td;
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
    int track_cnt;
};


class FeaturePerId
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int feature_id;
    int start_frame;
    deque<FeaturePerFrame> feature_per_frame; // 每一关键帧内的特征像素
    deque<eigen_q> w2c_q_vec; // 每一关键帧的姿态
    deque<vec_3> w2c_t_vec; // 每一关键帧的位置
    deque<FeaturePerFrame> feature_per_frame_all; // 特征点在所有帧中的像素
    deque<eigen_q> w2c_q_vec_all; // 特征点在所有帧的姿态
    deque<vec_3> w2c_t_vec_all; // 特征点在所有帧的位置



    int last_frame_id = 0;
    int current_frame_id = 0;

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    int not_see_time = 0; // 没有观测到的次数
    bool observed = false;

    std::deque<double> update_depth_delta; // 用于存储最近三次的更新量

    bool is_convegence = false; // 深度估计收敛
    bool triangulate_success = false;

    Vector3d gt_p;

    FeaturePerId(int _feature_id)  //以feature_id为索引，并保存了出现该角点的第一帧的id
        : feature_id(_feature_id), 
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
};


class FeatureManager{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FeatureManager(){}
    FeatureManager(Eigen::Matrix<double, 3, 3, 1, 3, 3>& intrinsic, int img_width_, int img_height_);
    void addFeatures(std::shared_ptr<Image_frame> img); // 添加特征
    void recoverFeature(std::shared_ptr<Image_frame> img, std::mutex& mtx, FeatureTracker& feature_track); // 恢复特征点
    void triangulate(); // 三角化
    void optimization(std::vector<RGB_pt_ptr> feature_pts); // 优化
    bool is_border(const cv::Point2f& pt); // 判断是否在图像边界
    /**
     * single level optical flow
     * @param [in] img1 the first image
     * @param [in] img2 the second image
     * @param [in] kp1 keypoints in img1
     * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
     * @param [out] success true if a keypoint is tracked successfully
     * @param [in] inverse use inverse formulation?
     */
    void OpticalFlowSingleLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const cv::Point2f &kp1,
        cv::Point2f &kp2,
        bool &success,
        bool inverse = false
    );

    float GetPixelValue(const cv::Mat &img, float x, float y);
    void calc_image_deriv_Sharr(std::vector<cv::Mat> &img_pyr,
                                std::vector<cv::Mat> &img_pyr_deriv_I,
                                std::vector<cv::Mat> &img_pyr_deriv_I_buff);
    void allocate_img_deriv_memory( std::vector<cv::Mat> &img_pyr,
                                    std::vector<cv::Mat> &img_pyr_deriv_I,
                                    std::vector<cv::Mat> &img_pyr_deriv_I_buff);

    std::list<FeaturePerId> feature;// 通过FeatureManager可以得到滑动窗口内所有的角点信息
    std::map<int, std::vector<std::vector<cv::Mat>>> pyr_images; // 特征点对应的图像，key为图像ID，value为金字塔图像、金字塔图像梯度、金字塔图像地图buffer
    // std::map<int, std::vector<cv::Mat>> pyr_images_deriv_I; // 强度梯度
    // std::map<int, std::vector<cv::Mat>> pyr_images_deriv_I_buffer; // 强度梯度buffer

    // static int oldest_frame_id;

    double parameter[1];
    std::map<int, RGB_pt_ptr> recover_feature_pts; // 恢复深度的特征点
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> intrinsic_inv;
    Eigen::Matrix2d f_intrinsic, f_intrinsic_inv;
    Eigen::Vector2d c_intrinsic, c_intrinsic_inv;

    bool inverse = false;
    // bool has_initial = true;

    int img_width;
    int img_height;

    cv::Size win_size = cv::Size(21, 21);
    int max_level = 3;

    double used_fov_margin = 0.005;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
    int flags = cv_OPTFLOW_USE_INITIAL_FLOW; // 使用初始值
    double minEigThreshold = 1e-4;
    
private:

};


#endif