// #include "projection_factor.h"

// Eigen::Matrix2d ProjectionFactor::sqrt_info = 450 / 1.5 * Eigen::Matrix2d::Identity();
// double ProjectionFactor::sum_t;

// ProjectionFactor::ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
//                                     const Eigen::Matrix3d& _Ri, const Eigen::Matrix3d& _Rj,
//                                     const Eigen::Vector3d& _Pi, const Eigen::Vector3d& _Pj) 
//                                     : pts_i(_pts_i), pts_j(_pts_j), Ri(_Ri), Rj(_Rj), Pi(_Pi), Pj(_Pj){};

// bool ProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
// {
//     // TicToc tic_toc;
//     // Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
//     // Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

//     // Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
//     // Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

//     // Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
//     // Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);


//     //pts_i 是i时刻归一化相机坐标系下的3D坐标
//     //第i帧相机坐标系下的的逆深度
//     double inv_dep_i = parameters[0][0];
//     // std::cout << "inv_dep_i " << inv_dep_i << std::endl;
//     //第i帧相机坐标系下的3D坐标
//     Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
//     // //第i帧IMU坐标系下的3D坐标
//     // Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
//     //世界坐标系下的3D坐标
//     Eigen::Vector3d pts_w = Ri * pts_camera_i + Pi;
//     //第j帧imu坐标系下的3D坐标
//     Eigen::Vector3d pts_camera_j = Rj.transpose() * (pts_w - Pj);
//     //第j帧相机坐标系下的3D坐标
//     // Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
//     Eigen::Map<Eigen::Vector2d> residual(residuals);


//     // 残差构建

//     double dep_j = pts_camera_j.z();
//     residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
//     // std::cout << "residual is " << residual << std::endl;

//     residual = sqrt_info * residual;
    

//     //reduce 表示残差residual对fci（pts_camera_j）的导数，同样根据不同的相机模型
//     if (jacobians)
//     {
//         // Eigen::Matrix3d Ri = Qi.toRotationMatrix();
//         // Eigen::Matrix3d Rj = Qj.toRotationMatrix();
//         // Eigen::Matrix3d ric = qic.toRotationMatrix();
//         Eigen::Matrix<double, 2, 3> reduce(2, 3);

//         reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
//             0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);

//         reduce = sqrt_info * reduce;

//         // 对逆深度 \lambda (inv_dep_i)
//         if (jacobians[0])
//         {
//             Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[0]);

//             jacobian_feature = reduce * Rj.transpose() * Ri * pts_i * -1.0 / (inv_dep_i * inv_dep_i);
//             std::cout << "jacobian_feature " << jacobian_feature << std::endl;

//         }
        
//     }
//     // sum_t += tic_toc.toc();

//     return true;
// }


