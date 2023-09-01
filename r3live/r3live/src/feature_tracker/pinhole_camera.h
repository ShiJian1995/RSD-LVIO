#ifndef PINHOLE_CAMERA_H
#define PINHOLE_CAMERA_H
#include <boost/shared_ptr.hpp>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

using namespace std;


class CameraParam{

public:
    CameraParam();
    CameraParam(int w, int h, int equalize,
                const Eigen::Matrix<double, 5, 1>& dist_coffes,
                const Eigen::Matrix3d& intrinsic);

    double& k1(void);
    double& k2(void);
    double& p1(void);
    double& p2(void);
    double& fx(void);
    double& fy(void);
    double& cx(void);
    double& cy(void);

    double k1(void) const;
    double k2(void) const;
    double p1(void) const;
    double p2(void) const;
    double fx(void) const;
    double fy(void) const;
    double cx(void) const;
    double cy(void) const;

    int& imageWidth(void);
    int& imageHeight(void);
    int& equalize(void);
    int imageWidth(void) const;
    int imageHeight(void) const;
    int equalize(void) const;
    int nIntrinsics(void) const;

    Eigen::Matrix<double, 5, 1> m_dist_coffes;
    Eigen::Matrix3d m_intrinsic;

private:
    double m_k1;
    double m_k2;
    double m_p1;
    double m_p2;
    double m_fx;
    double m_fy;
    double m_cx;
    double m_cy;

    int m_imageWidth;
    int m_imageHeight;
    int m_nIntrinsics;
    int EQUALIZE;
};

class PinholeCamera{

public:
    PinholeCamera();
    PinholeCamera(const CameraParam& params); // 有参构造函数
    const CameraParam& getParameters(void) const;

    // Lift points from the image plane to the projective space
    void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    void distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) const;

private:
    CameraParam camera_param;

    double m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23;
    bool m_noDistortion;

};

typedef boost::shared_ptr<PinholeCamera> PinholeCameraPtr;

#endif