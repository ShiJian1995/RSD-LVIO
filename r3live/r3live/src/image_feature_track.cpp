#include "r3live.hpp"

void R3LIVE::image_feature_track(){

    
    feature_tracker = FeatureTracker(pinhole_camera);  // 光流跟踪
    bool init_feature = false;
    const int WINDOW_SIZE = 20;
    imageFeature image_feature;
    // imageFeatureCloud image_feature_cloud; // 特征点集

    while(ros::ok()){

        ros::spinOnce(); // 执行回调函数（r3live中如此操作，后续调试该操作是否合理）
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) ); // 休眠1ms
        
        // for(auto img : m_queue_image_with_pose){
        if(!m_queue_image_with_pose.empty()){

            std::shared_ptr<Image_frame> img = m_queue_image_with_pose.front();
            if(!img->image_feature_cloud_ptr){

                
                std::chrono::time_point< std::chrono::system_clock > t1 = std::chrono::system_clock::now();
                
                feature_tracker.readImage(img);
                //更新全局ID
                for (unsigned int i = 0;; i++)
                {
                    bool completed = feature_tracker.updateID(i);
                    if (!completed)
                        break;
                }
                
                auto &un_pts = feature_tracker.cur_un_pts;
                auto &cur_pts = feature_tracker.cur_pts;
                auto &prev_pts = feature_tracker.prev_pts;
                auto &ids = feature_tracker.ids;
                auto &pts_velocity = feature_tracker.pts_velocity;
                auto &track_cnt = feature_tracker.track_cnt;
                
                m_camera_data_mutex.lock(); // 这样加锁是否正确？还是应该添加到for循环之前？
                img->image_feature_cloud_ptr = std::make_shared<std::vector<imageFeature>>(); // 特征点集

                for(auto j = 0; j < ids.size(); j++){ // 遍历特征点

                    if(feature_tracker.track_cnt[j] > 1){ // 跟踪次数足够多

                        image_feature.point_id = ids[j]; // ID
                        image_feature.point_cam = Eigen::Vector3d(un_pts[j].x, un_pts[j].y, 1); // 相机坐标系下的坐标
                        image_feature.point_img = Eigen::Vector2d(cur_pts[j].x, cur_pts[j].y); // 像素坐标，u，v
                        image_feature.last_point_img = Eigen::Vector2d(prev_pts[j].x, prev_pts[j].y); // 上一时刻像素坐标，u，v
                        image_feature.point_vel = Eigen::Vector2d(pts_velocity[j].x, pts_velocity[j].y); // 像素速度
                        image_feature.track_time = track_cnt[j];
                        (img->image_feature_cloud_ptr)->push_back(image_feature);
                    }
                }
                m_camera_data_mutex.unlock();

                
                std::chrono::duration<double> optical_track_duration =  std::chrono::system_clock::now() - t1;

                std::cout << "duration is " << optical_track_duration.count() << std::endl;

                // 超过一定时间或者特征点已经恢复
                while(optical_track_duration.count() < 0.05 && feature_tracker.recover_pts.empty()){

                    optical_track_duration =  std::chrono::system_clock::now() - t1;
                    std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) ); // 休眠1ms
                }

                // 恢复特征点
                m_feature_recover_mutex.lock();
                if(!feature_tracker.recover_pts.empty()){

                    std::cout << "feature_tracker.recover_pts size is " << feature_tracker.recover_pts.size() << std::endl;

                    feature_tracker.track_recover_feature();

                    // feature_tracker.recover_pts.clear();
                    // feature_tracker.recover_ids.clear();
                    // feature_tracker.recover_cnt.clear();
                    
                }
                m_feature_recover_mutex.unlock();

                feature_tracker.prev_img = feature_tracker.cur_img;
                feature_tracker.cur_img = feature_tracker.forw_img;

                if (!init_feature)//第一帧不发布
                {
                    init_feature = true;
                }
                else{

                    if (1)
                    {
                        // //////////////////////////////////////////////////////////////////////
                        cv::Mat show_img = img->m_img.clone();
                        // cv_img_mono_ptr = cv_bridge::cvtColor(cv_img_mono_ptr, sensor_msgs::image_encodings::BGR8);
                        // cv::Mat tmp_img = img->m_img.clone();
                        cv::cvtColor(show_img, show_img, CV_BGR2GRAY);
                        cv::cvtColor(show_img, show_img, CV_GRAY2RGB);
                        //显示追踪状态，越红越好，越蓝越不行
                        for (unsigned int j = 0; j < feature_tracker.cur_pts.size(); j++)
                        {
                            double len = std::min(1.0, 1.0 * feature_tracker.track_cnt[j] / WINDOW_SIZE);
                            cv::circle(show_img, feature_tracker.cur_pts[j], 4, cv::Scalar(0, 255 * (1 - len), 255 * len), 2);
                            //draw speed line
                            /*
                            Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                            Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                            Vector3d tmp_prev_un_pts;
                            tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                            tmp_prev_un_pts.z() = 1;
                            Vector2d tmp_prev_uv;
                            trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                            cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                            */
                            //char name[10];
                            //sprintf(name, "%d", trackerData[i].ids[j]);
                            //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                        }
                        
                        pub_recover_match.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", show_img));
                    }
                }
            }
        }   
    }
}