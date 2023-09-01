#include "pinhole_camera.h"

//----------------------------------------- Parameters -----------------------------------------
CameraParam::CameraParam()
 : m_k1(0.0)
 , m_k2(0.0)
 , m_p1(0.0)
 , m_p2(0.0)
 , m_fx(0.0)
 , m_fy(0.0)
 , m_cx(0.0)
 , m_cy(0.0){}

CameraParam::CameraParam(int w, int h, int equalize,
                        const Eigen::Matrix<double, 5, 1>& dist_coffes,
                        const Eigen::Matrix3d& intrinsic)
                        : m_imageWidth(w)
                        , m_imageHeight(h)
                        , EQUALIZE(equalize){
    m_k1 = dist_coffes(0);
    m_k2 = dist_coffes(1);
    m_p1 = dist_coffes(2);
    m_p2 = dist_coffes(3);
    m_fx = intrinsic(0, 0);
    m_fy = intrinsic(1, 1);
    m_cx = intrinsic(0, 2);
    m_cy = intrinsic(1, 2);
    m_dist_coffes = dist_coffes;
    m_intrinsic = intrinsic;
}

double&
CameraParam::k1(void)
{
    return m_k1;
}

double&
CameraParam::k2(void)
{
    return m_k2;
}

double&
CameraParam::p1(void)
{
    return m_p1;
}

double&
CameraParam::p2(void)
{
    return m_p2;
}

double&
CameraParam::fx(void)
{
    return m_fx;
}

double&
CameraParam::fy(void)
{
    return m_fy;
}

double&
CameraParam::cx(void)
{
    return m_cx;
}

double&
CameraParam::cy(void)
{
    return m_cy;
}

double
CameraParam::k1(void) const
{
    return m_k1;
}

double
CameraParam::k2(void) const
{
    return m_k2;
}

double
CameraParam::p1(void) const
{
    return m_p1;
}

double
CameraParam::p2(void) const
{
    return m_p2;
}

double
CameraParam::fx(void) const
{
    return m_fx;
}

double
CameraParam::fy(void) const
{
    return m_fy;
}

double
CameraParam::cx(void) const
{
    return m_cx;
}

double
CameraParam::cy(void) const
{
    return m_cy;
}

int&
CameraParam::imageWidth(void)
{
    return m_imageWidth;
}

int&
CameraParam::imageHeight(void)
{
    return m_imageHeight;
}

int&
CameraParam::equalize(void)
{
    return EQUALIZE;
}

int
CameraParam::imageWidth(void) const
{
    return m_imageWidth;
}

int
CameraParam::imageHeight(void) const
{
    return m_imageHeight;
}

int
CameraParam::equalize(void) const
{
    return EQUALIZE;
}

int
CameraParam::nIntrinsics(void) const
{
    return m_nIntrinsics;
}


//----------------------------------------- PinholeCamera -----------------------------------------
PinholeCamera::PinholeCamera()
 : m_inv_K11(1.0)
 , m_inv_K13(0.0)
 , m_inv_K22(1.0)
 , m_inv_K23(0.0)
 , m_noDistortion(true){}

 PinholeCamera::PinholeCamera(const CameraParam& params)
 : camera_param(params)
{
    if ((camera_param.k1() == 0.0) &&
        (camera_param.k2() == 0.0) &&
        (camera_param.p1() == 0.0) &&
        (camera_param.p2() == 0.0))
    {
        m_noDistortion = true;
    }
    else
    {
        m_noDistortion = false;
    }

    m_noDistortion = true; // R3LIVE中已经去过畸变

    // Inverse camera projection matrix parameters
    m_inv_K11 = 1.0 / camera_param.fx();
    m_inv_K13 = -camera_param.cx() / camera_param.fx();
    m_inv_K22 = 1.0 / camera_param.fy();
    m_inv_K23 = -camera_param.cy() / camera_param.fy();
}

const CameraParam&
PinholeCamera::getParameters(void) const
{
    return camera_param;
}

/**
 * \brief Lifts a point from the image plane to its projective ray
 *
 * \param p image coordinates
 * \param P coordinates of the projective ray
 */
void
PinholeCamera::liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const
{
    double mx_d, my_d,mx2_d, mxy_d, my2_d, mx_u, my_u;
    double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;
    //double lambda;

    // Lift points to normalised plane
    mx_d = m_inv_K11 * p(0) + m_inv_K13;
    my_d = m_inv_K22 * p(1) + m_inv_K23;

    if (m_noDistortion)
    {
        mx_u = mx_d;
        my_u = my_d;
    }
    else
    {
        if (0)
        {
            double k1 = camera_param.k1();
            double k2 = camera_param.k2();
            double p1 = camera_param.p1();
            double p2 = camera_param.p2();

            // Apply inverse distortion model
            // proposed by Heikkila
            mx2_d = mx_d*mx_d;
            my2_d = my_d*my_d;
            mxy_d = mx_d*my_d;
            rho2_d = mx2_d+my2_d;
            rho4_d = rho2_d*rho2_d;
            radDist_d = k1*rho2_d+k2*rho4_d;
            Dx_d = mx_d*radDist_d + p2*(rho2_d+2*mx2_d) + 2*p1*mxy_d;
            Dy_d = my_d*radDist_d + p1*(rho2_d+2*my2_d) + 2*p2*mxy_d;
            inv_denom_d = 1/(1+4*k1*rho2_d+6*k2*rho4_d+8*p1*my_d+8*p2*mx_d);

            mx_u = mx_d - inv_denom_d*Dx_d;
            my_u = my_d - inv_denom_d*Dy_d;
        }
        else
        {
            // Recursive distortion model
            int n = 8;
            Eigen::Vector2d d_u;
            distortion(Eigen::Vector2d(mx_d, my_d), d_u);
            // Approximate value
            mx_u = mx_d - d_u(0);
            my_u = my_d - d_u(1);

            for (int i = 1; i < n; ++i)
            {
                distortion(Eigen::Vector2d(mx_u, my_u), d_u);
                mx_u = mx_d - d_u(0);
                my_u = my_d - d_u(1);
            }
        }
    }

    // Obtain a projective ray
    P << mx_u, my_u, 1.0;
}

/**
 * \brief Apply distortion to input point (from the normalised plane)
 *
 * \param p_u undistorted coordinates of point on the normalised plane
 * \return to obtain the distorted point: p_d = p_u + d_u
 */
void
PinholeCamera::distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) const
{
    double k1 = camera_param.k1();
    double k2 = camera_param.k2();
    double p1 = camera_param.p1();
    double p2 = camera_param.p2();

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
           p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}