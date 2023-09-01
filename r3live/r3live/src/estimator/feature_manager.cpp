#include "feature_manager.h"

// int FeatureManager::oldest_frame_id = 0; // 类内的静态变量

FeatureManager::FeatureManager(Eigen::Matrix<double, 3, 3, 1, 3, 3>& intrinsic, int img_width_, int img_height_){

    f_intrinsic = intrinsic.block<2, 2>(0, 0);
    c_intrinsic = intrinsic.block<2, 1>(0, 2);
    intrinsic_inv = intrinsic.inverse();
    f_intrinsic_inv = intrinsic_inv.block<2, 2>(0, 0);
    c_intrinsic_inv = intrinsic_inv.block<2, 1>(0, 2);

    img_width = img_width_;
    img_height = img_height_;
}

void FeatureManager::addFeatures(std::shared_ptr<Image_frame> img){

    cv::Mat img_gray_clone = img->m_img_gray; // 浅拷贝

    // 构建图像金字塔
    std::vector<cv::Mat> pyr_img, pyr_img_deriv_I, pyr_img_deriv_I_buffer;
    opencv_buildOpticalFlowPyramid(img_gray_clone, pyr_img, win_size, max_level, false);
    // pyr_images.insert(std::pair<int, std::vector<cv::Mat>>(img->m_frame_idx, pyr_img)); // 所有图像帧都添加到队列中
    // std::vector<cv::Mat> pyr_img_deriv_I, pyr_img_deriv_I_buffer;
    calc_image_deriv_Sharr(pyr_img, pyr_img_deriv_I, pyr_img_deriv_I_buffer);
    std::vector<std::vector<cv::Mat>> pyr_data;
    pyr_data.resize(3);
    pyr_data[0].resize(max_level + 1);
    pyr_data[1].resize(max_level + 1);
    pyr_data[2].resize(max_level + 1);
    for(int i = 0; i <= max_level; i++){
        pyr_data[0][i] = pyr_img[i]; // clone?
        pyr_data[1][i] = pyr_img_deriv_I[i];
        pyr_data[2][i] = pyr_img_deriv_I_buffer[i];
    }
    pyr_images.insert(std::pair<int, std::vector<std::vector<cv::Mat>>>(img->m_frame_idx, pyr_data)); // 所有图像帧都添加到队列中

    // pyr_images_deriv_I.insert(std::pair<int, std::vector<cv::Mat>>(img->m_frame_idx, pyr_img_deriv_I));
    // pyr_images_deriv_I_buffer.insert(std::pair<int, std::vector<cv::Mat>>(img->m_frame_idx, pyr_img_deriv_I_buffer));
    // std::cout << "feature size is " << feature.size() << std::endl;

    for(auto& f : feature){
        f.observed = false; // 初始认定为未观测
        f.w2c_q_vec_all.push_back(img->m_pose_w2c_q);
        f.w2c_t_vec_all.push_back(img->m_pose_w2c_t);
        f.current_frame_id = img->m_frame_idx;
    }

    for(auto &pt : *(img->image_feature_cloud_ptr)){

        // int observed_feature = 0;

        FeaturePerFrame f_per_fra(pt);
        int feature_id = pt.point_id; // 特征点的ID
        auto it = std::find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it){

            return it.feature_id == feature_id;
        });

        if(it == feature.end()){ // 没有该特征点
             
            feature.push_back(FeaturePerId(feature_id));

            // 关键帧
            feature.back().feature_per_frame.push_back(f_per_fra);
            feature.back().w2c_q_vec.push_back(img->m_pose_w2c_q);
            feature.back().w2c_t_vec.push_back(img->m_pose_w2c_t);
            // 全部帧
            feature.back().feature_per_frame_all.push_back(f_per_fra);
            feature.back().w2c_q_vec_all.push_back(img->m_pose_w2c_q);
            feature.back().w2c_t_vec_all.push_back(img->m_pose_w2c_t);
            
            feature.back().observed = true;
            feature.back().last_frame_id = img->m_frame_idx;
            feature.back().current_frame_id = img->m_frame_idx;

        }
        else if(it->feature_id == feature_id){ // 包含该特征点

            it->observed = true;
            it->feature_per_frame_all.push_back(f_per_fra);

            // 检查视差，满足要求则添加为关键帧，否则不添加
            Eigen::Vector2d uv0 = it->feature_per_frame.back().uv; // 上一关键帧中的像素坐标
            Eigen::Vector2d uv1 = f_per_fra.uv; // 当前帧中的像素坐标
            double parallax = (uv1 - uv0).norm(); // 视差
            if(parallax > 10){ // 视差大于10个像素，添加该特征点

                it->feature_per_frame.push_back(f_per_fra);
                it->w2c_q_vec.push_back(img->m_pose_w2c_q);
                it->w2c_t_vec.push_back(img->m_pose_w2c_t);
                // 更新frame id
                it->last_frame_id = img->m_frame_idx; // 最后关键帧ID
            }
        }
    }

    // 遍历特征点，如果特征点长时间没有被观测，则删除
    for(auto f = feature.begin(); f != feature.end(); ){

        if(!f->observed){
            f->not_see_time++;
        }
        if(f->not_see_time > 2){ // 2 次没有观测

            auto it = recover_feature_pts.find(f->feature_id);
            if(it != recover_feature_pts.end()){
                
                recover_feature_pts.erase(it); // 清空特征点
            }

            f = feature.erase(f);
            
        }
        else{
            ++f;
        }
    }
}

void FeatureManager::calc_image_deriv_Sharr(std::vector<cv::Mat> &img_pyr,
                                            std::vector<cv::Mat> &img_pyr_deriv_I,
                                            std::vector<cv::Mat> &img_pyr_deriv_I_buff)
{
    if (img_pyr_deriv_I_buff.size() == 0 ||
        img_pyr_deriv_I_buff[0].size().width == 0 ||
        img_pyr_deriv_I_buff[0].size().height == 0)
    {
        allocate_img_deriv_memory(img_pyr, img_pyr_deriv_I, img_pyr_deriv_I_buff);
    }
    // Calculate Image derivative
    for (int level = max_level; level >= 0; level--)
    {
        cv::Size imgSize = img_pyr[level].size();
        cv::Mat _derivI(imgSize.height + win_size.height * 2,
                        imgSize.width + win_size.width * 2, img_pyr_deriv_I_buff[level].type(), img_pyr_deriv_I_buff[level].ptr());
        img_pyr_deriv_I[level] = _derivI(cv::Rect(win_size.width, win_size.height, imgSize.width, imgSize.height));
        calc_sharr_deriv(img_pyr[level], img_pyr_deriv_I[level]);
        cv::copyMakeBorder(img_pyr_deriv_I[level], _derivI, win_size.height, win_size.height, win_size.width, win_size.width, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED);
    }
}

void FeatureManager::allocate_img_deriv_memory(std::vector<cv::Mat> &img_pyr,
                                                std::vector<cv::Mat> &img_pyr_deriv_I,
                                                std::vector<cv::Mat> &img_pyr_deriv_I_buff)
{
    int derivDepth = cv::DataType<deriv_type>::depth;
    img_pyr_deriv_I.resize(img_pyr.size());
    img_pyr_deriv_I_buff.resize(img_pyr.size());
    for (int level = max_level; level >= 0; level--)
    {
        if (img_pyr_deriv_I_buff[level].cols == 0)
        {
            // dI/dx ~ Ix, dI/dy ~ Iy
            // Create the pyramid mat with add the padding.
            img_pyr_deriv_I_buff[level].create(img_pyr[level].rows + win_size.height * 2,
                                               img_pyr[level].cols + win_size.width * 2,
                                               CV_MAKETYPE(derivDepth, img_pyr[level].channels() * 2));
        }
    }
}

// 使用对极几何恢复特征点
void FeatureManager::recoverFeature(std::shared_ptr<Image_frame> img, std::mutex& mtx, FeatureTracker& feature_track){

    // std::vector<FeatureTrackFalse> recover_features, false_features;
    std::vector<cv::Point2f> recover_pts, false_pts;
    std::vector<int> recover_pts_id, recover_track_cnt;

    std::map<int, std::vector<std::vector<cv::Mat>>>::iterator pyr_img_it;
    std::vector<cv::Mat> pyr_img1, pyr_img2, pyr_img1_der_I, pyr_img2_der_I, pyr_img1_der_I_buffer, pyr_img2_der_I_buffer;
    pyr_img1.resize(max_level);
    int oldest_frame_id = -1;

    cv::Point2f pt1, pt2; // 特征点像素

    // 遍历特征点
    for(auto &it_per_id : feature){

        // 查找最旧的图像ID,即最早的特征帧
        if(oldest_frame_id < 0){

            oldest_frame_id = it_per_id.last_frame_id;
        }
        else{

            if(oldest_frame_id > it_per_id.last_frame_id){

                oldest_frame_id = it_per_id.last_frame_id;
            }
        }

        // 如果特征点没有被恢复深度且在当前帧中跟踪失败
        if((!it_per_id.is_convegence) && (!it_per_id.observed)){

            // 关键帧中的特征点数目
            int feature_size = it_per_id.feature_per_frame.size(); 

            // std::cout << "feature size is " << feature_size << std::endl;

            if(feature_size > 2){ // 关键帧数目大于阈值

                // 如果特征点在边界附近，则不进行特征恢复
                pt1 = cv::Point2f(it_per_id.feature_per_frame.back().uv.x(), 
                                    it_per_id.feature_per_frame.back().uv.y()); // 上一帧中的点
                
                if(is_border(pt1)){

                    continue;
                }

                // 1、求解矩阵
                std::vector<Eigen::Matrix3d> delta_t_R; // 矩阵队列
                delta_t_R.resize(feature_size); // 所有点与最后点之间的变换关系
                std::vector<double> epipolar_constraint;
                epipolar_constraint.resize(feature_size - 1); // 次新帧中的重投影误差
                            

                for(int i = 0; i < feature_size; i++){

                    Eigen::Matrix3d delta_q;
                    Eigen::Vector3d delta_t;
                    Eigen::Matrix3d delta_t_skew;    

                    delta_q = it_per_id.w2c_q_vec[i].inverse() * 
                                it_per_id.w2c_q_vec_all.back();
                    delta_t = it_per_id.w2c_q_vec[i].inverse() * 
                                (it_per_id.w2c_t_vec_all.back() - it_per_id.w2c_t_vec[i]);
                    
                    delta_t_skew << SKEW_SYM_MATRIX(delta_t);
                    delta_t_R[i] = delta_t_skew * delta_q;
                }

                double min_epi = INT_MAX;

                for(int i = 0; i < feature_size - 1; i++){

                    Eigen::Vector3d delta_t;
                    eigen_q delta_q;
                    Eigen::Matrix3d delta_t_skew;

                    delta_q = it_per_id.w2c_q_vec[i].inverse() * 
                                it_per_id.w2c_q_vec[feature_size - 1];
                    delta_t = it_per_id.w2c_q_vec[i].inverse() * 
                                (it_per_id.w2c_t_vec[feature_size - 1] - it_per_id.w2c_t_vec[i]);

                    delta_t_skew << SKEW_SYM_MATRIX(delta_t);

                    
                    epipolar_constraint[i] = std::fabs(it_per_id.feature_per_frame[i].point.transpose() * 
                                            delta_t_skew * delta_q.toRotationMatrix() * 
                                            it_per_id.feature_per_frame[feature_size - 1].point); // 重投影误差
                    if(min_epi > epipolar_constraint[i]){
                        min_epi = epipolar_constraint[i];
                    }
                    // std::cout << "epipolar_constraint[i] " << epipolar_constraint[i] << std::endl;
                }

                // 归一化
                for(int i = 0; i < epipolar_constraint.size(); i++){
                    
                    epipolar_constraint[i] /= min_epi;
                    epipolar_constraint[i]++;
                }

                // 2、构建最小二乘
                Eigen::Matrix< double, -1, -1 > A_mat;
                A_mat.resize(feature_size, 2);
                A_mat.setZero();
                Eigen::Matrix<double, -1, -1> y_mat;
                y_mat.resize(feature_size, 1);
                y_mat.setZero();
                Eigen::Matrix< double, -1, -1 > weight_mat;
                weight_mat.resize(feature_size, feature_size);
                weight_mat.setIdentity();

                // 权重矩阵更新，根据向次新帧中的重投影误差设置权重矩阵
                for(int i = 0; i < feature_size; i++){
                    if(i < feature_size - 1)
                        weight_mat(i, i) = 1.0 / epipolar_constraint[i];
                    Eigen::Matrix<double, 1, 3> temp1 = it_per_id.feature_per_frame[i].point.transpose() * delta_t_R[i];
                    A_mat.block(i, 0, 1, 2) = temp1.head(2);
                    y_mat.block(i, 0, 1, 1) = -temp1.tail(1);
                }

                // std::cout << "weight_mat " << weight_mat << std::endl;

                Eigen::Vector2d temp2 = (A_mat.transpose() * weight_mat * A_mat).inverse() * (A_mat.transpose() * weight_mat * y_mat);
                Eigen::Vector2d temp3 = f_intrinsic * temp2 + c_intrinsic;

                // 1、确定图像、图像梯度和像素点坐标
                pyr_img_it = pyr_images.find(it_per_id.last_frame_id);
                if(pyr_img_it == pyr_images.end()){
                    ROS_ERROR_STREAM("error images !!!!");
                }
                else{

                    pyr_img1 = pyr_img_it->second[0];
                    pyr_img1_der_I = pyr_img_it->second[1];
                    pyr_img1_der_I_buffer = pyr_img_it->second[2];

                }

                pyr_img_it = pyr_images.find(it_per_id.current_frame_id);
                if(pyr_img_it == pyr_images.end()){
                    ROS_ERROR_STREAM("error images !!!!");
                }
                else{
                    pyr_img2 = pyr_img_it->second[0];
                    pyr_img2_der_I = pyr_img_it->second[1];
                    pyr_img2_der_I_buffer = pyr_img_it->second[2];

                }

                
                pt2 = cv::Point2f(temp3.x(), temp3.y()); // 当前帧中的点

                // std::cout << "pt2 is " << pt2 << std::endl;

                if(pt2.x < 0 || pt2.y < 0 || pt2.x > img_width || pt2.y > img_height || 
                    std::isnan(pt2.x) || std::isnan(pt2.y)){ // 如果预测的特征点像素坐标小于阈值
                    continue;
                }

                // 2、多层光流跟踪

                bool success = true;
                cv::Point2f pt2_final = pt2;

                std::vector<cv::Point2f> pt1_vec;
                pt1_vec.push_back(pt1);
                std::vector<cv::Point2f> pt2_vec;
                pt2_vec.push_back(pt2);
                std::vector<uchar> status;
                status.resize(1);
                status[0] = 1;
                std::vector<float> err;
                err.resize(1);

                for(int level = max_level; level >= 0; level--){

                    calculate_LK_optical_flow(cv::Range(0, 1), &pyr_img1[level], &pyr_img1_der_I[level],
                                                &pyr_img2[level], pt1_vec.data(),
                                                pt2_vec.data(), status.data(), err.data(), win_size,
                                                criteria, level, max_level, flags,
                                                minEigThreshold);
                }
                pt2 = pt2_vec.front();


                if(0){

                    cv::Mat show_img1 = pyr_img1[0].clone();
                    // cv_img_mono_ptr = cv_bridge::cvtColor(cv_img_mono_ptr, sensor_msgs::image_encodings::BGR8);
                    // cv::Mat tmp_img = img->m_img.clone();
                    // cv::cvtColor(show_img, show_img, CV_BGR2GRAY);
                    cv::cvtColor(show_img1, show_img1, CV_GRAY2RGB);
                    //显示追踪状态，越红越好，越蓝越不行
                    
                    cv::circle(show_img1, pt1, 4, cv::Scalar(0, 0, 255), 2);
                    cv::imshow("last",show_img1);
                    cv::waitKey(1);

                    cv::Mat show_img2 = pyr_img2[0].clone();
                    // cv::cvtColor(show_img1, show_img1, CV_BGR2GRAY);
                    cv::cvtColor(show_img2, show_img2, CV_GRAY2RGB);
                    
                    cv::circle(show_img2, pt2, 4, cv::Scalar(255, 0, 0), 2); // 光流跟踪，蓝色
                    cv::circle(show_img2, pt2_final, 4, cv::Scalar(0, 255, 0), 2); // 预测， 绿色
                    cv::circle(show_img2, pt1, 4, cv::Scalar(0, 0, 255), 2); // 原始点，红色
                    cv::imshow("current",show_img2);
                    cv::waitKey(1);

                }

                if(status.front()){ // 跟踪成功

                    // if(std::fabs((pt2_final - pt2).x) < 20 && std::fabs((pt2_final - pt2).y) < 20){ // 光流跟踪异常

                    //     pt2_final = pt2; // 如果最后优化正确，则使用优化结果，否则使用预测值
                    // }
                    
                    pt2_final = pt2;
                }

                // std::cout << "update " << pt2_final << std::endl;


                recover_pts.push_back(pt2_final);
                false_pts.push_back(pt1);
                recover_pts_id.push_back(it_per_id.feature_id);
                recover_track_cnt.push_back(it_per_id.feature_per_frame_all.back().track_cnt);

            }
        }
    }

    // 恢复特征点
    mtx.lock();

    // if(!img->recovered_feature_ptr){ // 空指针

        // img->recovered_feature_ptr = std::make_shared<std::vector<imageFeature>>(); // 分配内存

        // std::cout << "recover_pts.size() " << recover_pts.size() << std::endl;


            // imageFeature img_f;
            // img_f.last_point_img = Eigen::Vector2d(false_pts[i].x, false_pts[i].y);
            // img_f.point_id = recover_pts_id[i];
            // img_f.track_time = recover_track_cnt[i];
            // img_f.point_img = Eigen::Vector2d(recover_pts[i].x, recover_pts[i].y);
            // (img->recovered_feature_ptr)->push_back(img_f);

        feature_track.add_recover_feature(recover_pts_id, recover_track_cnt, recover_pts);

        
    // }
    mtx.unlock();

    // 删除更早的图像ID
    for(auto img = pyr_images.begin(); img != pyr_images.end(); ){

        if(img->first < oldest_frame_id){

            img = pyr_images.erase(img);
        }
        else{
            img++;
        }
    }

}

bool FeatureManager::is_border(const cv::Point2f& pt){


    if ((pt.x >= (used_fov_margin * img_width + 1)) && (std::ceil(pt.x) < ((1 - used_fov_margin) * img_width)) &&
        (pt.x >= (used_fov_margin * img_height + 1)) && (std::ceil(pt.y) < ((1 - used_fov_margin) * img_height))){

        return false;
    }
    else{
        return true;
    }
}

void FeatureManager::triangulate(){

    for(auto &it_per_id : feature){ // 对每一个特征点进行三角化

        if(it_per_id.estimated_depth > 0.1 ){ // 已经三角化

            continue;
        }

        int feature_size = it_per_id.feature_per_frame.size();
        // std::cout << "feature size is " << feature_size << std::endl;
        if(feature_size < 5){ // 特征点数据量较小，不能进行三角化恢复深度

            continue;
        }

        Eigen::MatrixXd svd_A(2 * feature_size, 4);

        // 第一帧相机坐标系相对世界坐标系
        Eigen::Matrix3d R0 = it_per_id.w2c_q_vec.front().toRotationMatrix();
        Eigen::Vector3d t0 = it_per_id.w2c_t_vec.front();

        for(int i = 0; i < feature_size; i++){ // 遍历之前的所有特征点

            Eigen::Matrix3d R1 = it_per_id.w2c_q_vec[i].toRotationMatrix();
            Eigen::Vector3d t1 = it_per_id.w2c_t_vec[i];
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);

            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_id.feature_per_frame[i].point.normalized();
            svd_A.row(i) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(i) = f[1] * P.row(2) - f[2] * P.row(1);

        }
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        if(std::fabs(svd_V[3]) < 1e-5){
            
            it_per_id.estimated_depth = 5; // 如果估计不准确，则假定一个初始值
            continue;
        }
        double svd_method = svd_V[2] / svd_V[3];
        it_per_id.estimated_depth = svd_method;
        if(it_per_id.estimated_depth < 0.1 || !isfinite(it_per_id.estimated_depth) ){

            it_per_id.estimated_depth = 5; // 如果估计不准确，则假定一个初始值
        }

    }
}

void FeatureManager::optimization(std::vector<RGB_pt_ptr> feature_pts){

    feature_pts.clear(); // 清空特征点
    
    for(auto& pt_per_id : feature){

        if(pt_per_id.estimated_depth < 0.1){ // 深度没有初始化
            continue;
        }

        if(!pt_per_id.observed){
            continue;
        }

        if(pt_per_id.feature_per_frame.size() < 10){

            continue;
        }

        // 收敛则不再进行估计
        if(pt_per_id.is_convegence){ 

            // 更新特征点的像素值
            auto it = recover_feature_pts.find(pt_per_id.feature_id);
            if(it != recover_feature_pts.end()){

                (it->second)->m_img_pt_in_current_frame = pt_per_id.feature_per_frame.back().uv; // 更新像素值
            }
            else{
                ROS_ERROR_STREAM("ERROR feature");
            }

            continue;
        }

        ceres::Problem::Options problem_options;
        ceres::Problem problem(problem_options);
        ceres::LossFunction *loss_function;
        loss_function = new ceres::HuberLoss(1.0);
        // loss_function = new ceres::CauchyLoss(1.0);
  
        parameter[0] = 1.0 / pt_per_id.estimated_depth;
        problem.AddParameterBlock(parameter, 1);

        Eigen::Vector3d pts_i = pt_per_id.feature_per_frame[0].point;
        int pt_size = pt_per_id.feature_per_frame.size();
        eigen_q Q_i = pt_per_id.w2c_q_vec[0];
        Eigen::Vector3d P_i = pt_per_id.w2c_t_vec[0];
        for(int i = 1; i < pt_size; i++){

            Eigen::Vector3d pts_j = pt_per_id.feature_per_frame[i].point;
            eigen_q Q_j = pt_per_id.w2c_q_vec[i];
            Eigen::Vector3d P_j = pt_per_id.w2c_t_vec[i];
            // ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j, R_i, R_j, P_i, P_j);
            ceres::CostFunction *cost_function = FeatureDepthFactor::Create(pts_i, pts_j, Q_i, Q_j, P_i, P_j);

            problem.AddResidualBlock(cost_function, loss_function, parameter);
        }

        ceres::Solver::Options options;

        options.linear_solver_type = ceres::DENSE_QR;
        options.num_threads = 2;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.max_num_iterations = 10;

        options.max_solver_time_in_seconds = 0.02;
        options.check_gradients = false;
        options.gradient_check_relative_precision = 1e-4;

        ceres::Solver::Summary summary;
        
        ceres::Solve(options, &problem, &summary);
        // std::cout << summary.BriefReport() << "\n";

        pt_per_id.update_depth_delta.push_back(std::fabs(1.0 / parameter[0] - pt_per_id.estimated_depth)); // 添加更新值

        pt_per_id.estimated_depth = 1.0 / parameter[0];

        // 滑动窗口
        Eigen::Vector3d pt_w = Q_i * (pts_i * pt_per_id.estimated_depth) + P_i; // 世界坐标下坐标
        eigen_q Q_i1 = pt_per_id.w2c_q_vec[1];
        Eigen::Vector3d P_i1 = pt_per_id.w2c_t_vec[1];
        Eigen::Vector3d pt_camera_i1 = Q_i1.inverse() * (pt_w - P_i1);

        pt_per_id.estimated_depth = pt_camera_i1.z();

        if(pt_per_id.update_depth_delta.size() > 1){

            bool fullfill = true;
            for(auto d : pt_per_id.update_depth_delta){

                if(d > 0.1){ // 深度变化大于5cm
                    fullfill = false;
                }
            }

            // 满足收敛添加，添加到地图中
            if(fullfill){ 
                pt_per_id.is_convegence = true;

                std::shared_ptr<RGB_pts> pt_rgb = std::make_shared<RGB_pts>();
                pt_rgb->set_pos(pt_w);
                pt_rgb->m_img_pt_in_current_frame = pt_per_id.feature_per_frame.back().uv; // 最后点为当前图像帧中的特征点
                feature_pts.push_back(pt_rgb);

                recover_feature_pts.insert(std::pair<int, RGB_pt_ptr>(pt_per_id.feature_id, pt_rgb));

            }
            pt_per_id.update_depth_delta.pop_front();
        }

        pt_per_id.feature_per_frame.pop_front();
        pt_per_id.w2c_q_vec.pop_front();
        pt_per_id.w2c_t_vec.pop_front(); 

    }

}

void FeatureManager::OpticalFlowSingleLevel(const cv::Mat &img1, const cv::Mat &img2,
                            const cv::Point2f &kp1, cv::Point2f &kp2,
                            bool &success,
                            bool inverse){

    int half_patch_size = 4;
    int iterations = 10;

    double dx = 0, dy = 0;
    dx = kp2.x - kp1.x;
    dy = kp2.y - kp1.y; // 待估计值

    double cost = 0, lastCost = 0;
    success = true; 

    // Gauss-Newton iterations
    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // hessian
    Eigen::Vector2d b = Eigen::Vector2d::Zero();    // bias
    Eigen::Vector2d J;  // jacobian

    for (int iter = 0; iter < iterations; iter++) {

        if (inverse == false) {
            H = Eigen::Matrix2d::Zero();
            b = Eigen::Vector2d::Zero();
        } else {
            // only reset b
            b = Eigen::Vector2d::Zero();
        }

        cost = 0;

        // compute cost and jacobian
        for (int x = -half_patch_size; x < half_patch_size; x++)
            for (int y = -half_patch_size; y < half_patch_size; y++) {
                double error = GetPixelValue(img1, kp1.x + x, kp1.y + y) -
                                GetPixelValue(img2, kp1.x + x + dx, kp1.y + y + dy);  // Jacobian
                if (inverse == false) {
                    J = -1.0 * Eigen::Vector2d(
                        0.5 * (GetPixelValue(img2, kp1.x + dx + x + 1, kp1.y + dy + y) -
                                GetPixelValue(img2, kp1.x + dx + x - 1, kp1.y + dy + y)),
                        0.5 * (GetPixelValue(img2, kp1.x + dx + x, kp1.y + dy + y + 1) -
                                GetPixelValue(img2, kp1.x + dx + x, kp1.y + dy + y - 1))
                    );
                } else if (iter == 0) {
                    // in inverse mode, J keeps same for all iterations
                    // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                    J = -1.0 * Eigen::Vector2d(
                        0.5 * (GetPixelValue(img1, kp1.x + x + 1, kp1.y + y) -
                                GetPixelValue(img1, kp1.x + x - 1, kp1.y + y)),
                        0.5 * (GetPixelValue(img1, kp1.x + x, kp1.y + y + 1) -
                                GetPixelValue(img1, kp1.x + x, kp1.y + y - 1))
                    );
                }
                // compute H, b and set cost;
                b += -error * J;
                cost += error * error;
                if (inverse == false || iter == 0) {
                    // also update H
                    H += J * J.transpose();
                }
            }

        // compute update
        Eigen::Vector2d update = H.ldlt().solve(b);

        if (std::isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            success = false;
            break;
        }

        if (iter > 0 && cost > lastCost) {
            break;
        }

        // update dx, dy
        dx += update[0];
        dy += update[1];
        lastCost = cost;
        success = true;

        if (update.norm() < 1e-2) {
            // converge
            break;
        }
    }

    kp2 += cv::Point2f(dx, dy); // 更新
}

/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */

float FeatureManager::GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;
    
    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}