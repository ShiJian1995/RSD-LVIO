#pragma once

#include <ros/ros.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "image_frame.hpp"

// class ProjectionFactor : public ceres::SizedCostFunction<1, 1>
// {
//   public:
//     ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
//                       const Eigen::Matrix3d& _Ri, const Eigen::Matrix3d& _Rj,
//                       const Eigen::Vector3d& _Pi, const Eigen::Vector3d& _Pj);
//     virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
//     // void check(double **parameters);

//     Eigen::Vector3d pts_i, pts_j;
//     Eigen::Vector3d Pi, Pj; // 位置
//     Eigen::Matrix3d Ri, Rj; // 姿态
//     Eigen::Matrix<double, 2, 3> tangent_base;
//     static Eigen::Matrix2d sqrt_info;
//     static double sum_t;
// };

struct FeatureDepthFactor{

  FeatureDepthFactor(const Eigen::Vector3d _pts_i, const Eigen::Vector3d _pts_j,
                      const eigen_q _Qi, const eigen_q _Qj,
                      const Eigen::Vector3d _Pi, const Eigen::Vector3d _Pj)
                      : pts_i(_pts_i), pts_j(_pts_j), Qi(_Qi), Qj(_Qj), Pi(_Pi), Pj(_Pj){}

  template <typename T> 
  bool operator()(const T* depth, T* residual) const{

    // std::cout << "pts_i " << pts_i << std::endl;

    // std::cout << "pts_i x  " << T(pts_i.x()) << std::endl;

    Eigen::Matrix<T, 3, 1> pti{T(pts_i.x()), T(pts_i.y()), T(pts_i.z())};
    Eigen::Matrix<T, 3, 1> ptj{T(pts_j.x()), T(pts_j.y()), T(pts_j.z())};
    Eigen::Matrix<T, 3, 1> pi{T(Pi.x()), T(Pi.y()), T(Pi.z())};
    Eigen::Matrix<T, 3, 1> pj{T(Pj.x()), T(Pj.y()), T(Pj.z())};
    Eigen::Quaternion<T> qi{T(Qi.w()), T(Qi.x()), T(Qi.y()), T(Qi.z())};
    Eigen::Quaternion<T> qj{T(Qj.w()), T(Qj.x()), T(Qj.y()), T(Qj.z())};
    
    // std::cout << "pti " << pti << std::endl;
    // std::cout << "ptj " << ptj << std::endl;
    // std::cout << "(depth[0]) " << (depth[0]) << std::endl;

    Eigen::Matrix<T, 3, 1> pts_camera_i = pti / (depth[0]);
    // std::cout << "x[0] " << depth[0] << std::endl;
    // std::cout << "pts_camera_i " << pts_camera_i << std::endl;
    Eigen::Matrix<T, 3, 1> pts_w = qi * pts_camera_i + pi;
    // std::cout << "pts_w " << pts_w << std::endl;

    // std::cout << "Qjw is " << Qj.w() << std::endl;
    // std::cout << "Qjx is " << Qj.x() << std::endl;
    // std::cout << "Qjy is " << Qj.y() << std::endl;
    // std::cout << "Qjz is " << Qj.z() << std::endl;

    
    //第j帧imu坐标系下的3D坐标
    Eigen::Matrix<T, 3, 1> pts_camera_j = qj.inverse() * (pts_w - pj);
    // Eigen::Map<Eigen::Vector2d> residuals(residual);
    // Eigen::Matrix<T, 3, 1> pc_j{T(pts_camera_j.x()), T(pts_camera_j.y()), T(pts_camera_j.z())};


    // 残差构建

    T dep_j = pts_camera_j.z();
    // Eigen::Matrix<T, 2, 1> res = (pts_camera_j / dep_j).head<2>() - ptj.head<2>();

    // std::cout << "pts_camera_j " <<  pts_camera_j << std::endl;
    // std::cout << "dep_j " <<  dep_j << std::endl;
    // std::cout << "ptj " <<  ptj << std::endl;

    residual[0] = (pts_camera_j / dep_j - ptj).x();
    residual[1] = (pts_camera_j / dep_j - ptj).y();
    // std::cout << "(pts_camera_j / dep_j - ptj).x() " << (pts_camera_j / dep_j - ptj).x() << std::endl;
    // std::cout << "(pts_camera_j / dep_j - ptj).y() " << (pts_camera_j / dep_j - ptj).y() << std::endl;
    // std::cout << "residual[0] " << residual[0] << std::endl;
    // std::cout << "residual[1] " << residual[1] << std::endl;

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d _pts_i, const Eigen::Vector3d _pts_j,
                                      const eigen_q _Qi, const eigen_q _Qj,
                                      const Eigen::Vector3d _Pi, const Eigen::Vector3d _Pj){

    return (new ceres::AutoDiffCostFunction<FeatureDepthFactor, 2, 1>
              (new FeatureDepthFactor(_pts_i, _pts_j, _Qi, _Qj, _Pi, _Pj)));

  }


  Eigen::Vector3d pts_i, pts_j;
  Eigen::Vector3d Pi, Pj; // 位置
  eigen_q Qi, Qj; // 姿态

};
