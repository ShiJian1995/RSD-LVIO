#ifndef FEATURE_TRACKER_H
#define FEATURE_TRACKER_H
// #include "common_lib.h"
#include "pinhole_camera.h"
#include "image_frame.hpp"

// 跟踪失败
struct FeatureTrackFalse{

    cv::Point2f cur_pts; // 跟踪失败点的像素坐标
    int ids; // 跟踪失败特征点的ID
    int track_cnt; // 失败的点已经被跟踪次数
};

/**
* @class FeatureTracker
* @Description 视觉前端预处理：对特征点进行光流跟踪;
*/

class FeatureTracker{

    public:

    FeatureTracker();
    FeatureTracker(PinholeCameraPtr pinhole_camera);
    void readImage(const std::shared_ptr<Image_frame>& image_frame);
    bool inBorder(const cv::Point2f &pt);
    void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status);
    void reduceVector(std::vector<int> &v, std::vector<uchar> status);
    void rejectWithF();
    void setMask();
    void addPoints();
    void undistortedPoints();
    bool updateID(unsigned int i);
    void add_recover_feature(const std::vector<int>& id, const std::vector<int>& cnt, const std::vector<cv::Point2f>& rec_pt);
    void track_recover_feature(); 
    void pt_img2cam(const cv::Point2f& pt_img, cv::Point2f& pt_cam);
    FeatureTracker& operator=(const FeatureTracker& other){

        m_pinhole_camera = other.m_pinhole_camera;
        m_cam_param = other.m_cam_param; // 参数赋值

        ROW = other.ROW;
        COL = other.COL;
        return *this;
    }

    PinholeCameraPtr m_pinhole_camera;
    CameraParam m_cam_param;
    int COL, ROW;
    double cur_time; // 当前跟踪图像的时间
    double prev_time; // 上一帧图像时间

    // prev_img是上一次发布的帧的图像数据
    // cur_img是光流跟踪的前一帧的图像数据
    // forw_img是光流跟踪的后一帧的图像数据
    cv::Mat prev_img, cur_img, forw_img;
    std::vector<cv::Point2f> n_pts;//每一帧中新提取的特征点
    std::vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    std::vector<cv::Point2f> prev_un_pts, cur_un_pts;//归一化相机坐标系下的坐标
    std::vector<int> ids;//能够被跟踪到的特征点的id
    std::vector<int> track_cnt;//当前帧forw_img中每个特征点被追踪的时间次数
    const double FOCAL_LENGTH = 460.0; // 虚假的焦距
    const double F_THRESHOLD = 1.0; // ransac算法的门限
    const int MIN_DIST = 100; // 特征点选择邻域阈值
    const int MAX_CNT = 100; // 特征点最大数目
    cv::Mat mask;//图像掩码
    std::map<int, cv::Point2f> cur_un_pts_map;
    std::map<int, cv::Point2f> prev_un_pts_map;
    std::vector<cv::Point2f> pts_velocity; //当前帧相对前一帧特征点沿x,y方向的像素移动速度
    static int n_id;//特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1

    std::vector<cv::Point2f> recover_pts; // 恢复的特征点
    std::vector<int> recover_ids; // 恢复的特征点的ID
    std::vector<int> recover_cnt; // 恢复特征点的跟踪时长
};

#endif