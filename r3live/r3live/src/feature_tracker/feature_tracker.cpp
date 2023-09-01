#include "feature_tracker.h"
//FeatureTracker的static成员变量n_id初始化为0
int FeatureTracker::n_id = 0;

//空的构造函数
FeatureTracker::FeatureTracker(){}

FeatureTracker::FeatureTracker(PinholeCameraPtr pinhole_camera)
{
    // cam_mod = _cam_mod;
    m_pinhole_camera = pinhole_camera;
    m_cam_param = pinhole_camera->getParameters(); // 参数赋值

    ROW = m_cam_param.imageHeight();
    COL = m_cam_param.imageWidth();
}

/**
 * @brief   对图像使用光流法进行特征点跟踪
 * @Description createCLAHE() 对图像进行自适应直方图均衡化
 *              calcOpticalFlowPyrLK() LK金字塔光流法
 *              setMask() 对跟踪点进行排序，设置mask
 *              rejectWithF() 通过基本矩阵剔除outliers
 *              goodFeaturesToTrack() 添加特征点(shi-tomasi角点)，确保每帧都有足够的特征点
 *              addPoints()添加新的追踪点
 *              undistortedPoints() 对角点图像坐标去畸变矫正，并计算每个角点的速度
 * @param[in]   _img 输入图像
 * @param[in]   _cur_time 当前时间（图像时间戳）
 * @return      void
*/
void FeatureTracker::readImage(const std::shared_ptr<Image_frame>& image_frame){

    cv::Mat img;
    cur_time = image_frame->m_timestamp;

    //如果EQUALIZE=1，表示太亮或太暗，进行直方图均衡化处理
    if (m_cam_param.equalize())
    {
        //自适应直方图均衡
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(image_frame->m_img_gray, img);
    }
    else
        img = image_frame->m_img_gray;

    if (forw_img.empty())
    {
        //如果当前帧的图像数据forw_img为空，说明当前是第一次读入图像数据
        //将读入的图像赋给当前帧forw_img，同时还赋给prev_img、cur_img
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        //否则，说明之前就已经有图像读入，只需要更新当前帧forw_img的数据
        forw_img = img;
    }

    //此时forw_pts还保存的是上一帧图像中的特征点，所以把它清除
    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        // TicToc t_o;
        std::vector<uchar> status;
        std::vector<float> err;

        //调用cv::calcOpticalFlowPyrLK()对前一帧的特征点cur_pts进行LK金字塔光流跟踪，得到forw_pts
        //status标记了从前一帧cur_img到forw_img特征点的跟踪状态，无法被追踪到的点标记为0
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        //将位于图像边界外的点标记为0
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;

        //根据status,把跟踪失败的点剔除
        //不仅要从当前帧数据forw_pts中剔除，而且还要从cur_un_pts、prev_pts和cur_pts中剔除
        //prev_pts和cur_pts中的特征点是一一对应的
        //记录特征点id的ids，和记录特征点被跟踪次数的track_cnt也要剔除
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        // ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    //光流追踪成功,特征点被成功跟踪的次数就加1
    //数值代表被追踪的次数，数值越大，说明被追踪的就越久
    for (auto &n : track_cnt)
        n++;
    
    //通过本质矩阵剔除outliers
    rejectWithF();
    setMask();//保证相邻的特征点之间要相隔30个像素,设置mask
    //计算是否需要提取新的特征点
    int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0)
    {
        if(mask.empty())
            std::cout << "mask is empty " << std::endl;
        if (mask.type() != CV_8UC1)
            std::cout << "mask type wrong " << std::endl;
        if (mask.size() != forw_img.size())
            std::cout << "wrong size " << std::endl;
        /** 
         *void cv::goodFeaturesToTrack(    在mask中不为0的区域检测新的特征点
            *   InputArray  image,              输入图像
            *   OutputArray     corners,        存放检测到的角点的vector
            *   int     maxCorners,             返回的角点的数量的最大值
            *   double  qualityLevel,           角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
            *   double  minDistance,            返回角点之间欧式距离的最小值
            *   InputArray  mask = noArray(),   和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
            *   int     blockSize = 3,          计算协方差矩阵时的窗口大小
            *   bool    useHarrisDetector = false,  指示是否使用Harris角点检测，如不指定则使用shi-tomasi算法
            *   double  k = 0.04                Harris角点检测需要的k值
            *)   
            */
        cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
    }
    else
        n_pts.clear();

    //添将新检测到的特征点n_pts添加到forw_pts中，id初始化-1,track_cnt初始化为1.
    addPoints();

    //当下一帧图像到来时，当前帧数据就成为了上一帧发布的数据
    
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;

    //把当前帧的数据forw_img、forw_pts赋给上一帧cur_img、cur_pts
    
    cur_pts = forw_pts;

    //根据不同的相机模型去畸变矫正和转换到归一化坐标系上，计算速度
    undistortedPoints();
    prev_time = cur_time;
}

//判断跟踪的特征点是否在图像边界内
bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    //cvRound()：返回跟参数最接近的整数值，即四舍五入；
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

//去除无法跟踪的特征点
void FeatureTracker::reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

//去除无法追踪到的特征点
void FeatureTracker::reduceVector(std::vector<int> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

/**
 * @brief   通过F矩阵去除outliers
 * @Description 将图像坐标转换为归一化坐标
 *              cv::findFundamentalMat()计算F矩阵
 *              reduceVector()去除outliers 
 * @return      void
*/
void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        // ROS_DEBUG("FM ransac begins");
        // TicToc t_f;

        std::vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {

            Eigen::Vector3d tmp_p;
            //根据不同的相机模型将二维坐标转换到三维坐标
            m_pinhole_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            //转换为归一化像素坐标
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_pinhole_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        std::vector<uchar> status;
        //调用cv::findFundamentalMat对un_cur_pts和un_forw_pts计算F矩阵
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        // int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        // ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        // ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

/**
 * @brief   对跟踪点进行排序并去除密集点
 * @Description 对跟踪到的特征点，按照被追踪到的次数排序并依次选点
 *              使用mask进行类似非极大抑制，半径为30，去掉密集点，使特征点分布均匀            
 * @return      void
*/
void FeatureTracker::setMask()
{

    mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    
    // prefer to keep features that are tracked for long time
    // 构造(cnt，pts，id)序列
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(std::make_pair(track_cnt[i], std::make_pair(forw_pts[i], ids[i])));

    //对光流跟踪到的特征点forw_pts，按照被跟踪到的次数cnt从大到小排序（lambda表达式）
    std::sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const std::pair<int, std::pair<cv::Point2f, int>> &a, const std::pair<int, std::pair<cv::Point2f, int>> &b)
    {
        return a.first > b.first;
    });

    //清空cnt，pts，id并重新存入
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            //当前特征点位置对应的mask值为255，则保留当前特征点，将对应的特征点位置pts，id，被追踪次数cnt分别存入
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);

            //在mask中将当前特征点周围半径为MIN_DIST的区域设置为0，后面不再选取该区域内的点（使跟踪点不集中在一个区域上）
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

//添将新检测到的特征点n_pts
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);//新提取的特征点id初始化为-1
        track_cnt.push_back(1);//新提取的特征点被跟踪的次数初始化为1
    }
}


//对角点图像坐标进行去畸变矫正，转换到归一化坐标系上，并计算每个角点的速度。                       
void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());

    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;

        //根据不同的相机模型将二维坐标转换到三维坐标
        m_pinhole_camera->liftProjective(a, b);

        //再延伸到深度归一化平面上
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }

    // 计算每个特征点的速度到pts_velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}

//更新特征点id
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::add_recover_feature(const std::vector<int>& id, const std::vector<int>& cnt, 
                                    const std::vector<cv::Point2f>& rec_pt){ // 恢复跟踪的特征点

    recover_pts.clear();
    recover_ids.clear();
    recover_cnt.clear();
    for(int i = 0; i < rec_pt.size(); i++){
        
        recover_pts.push_back(rec_pt[i]);
        recover_ids.push_back(id[i]);
        recover_cnt.push_back(cnt[i]);
    }
}

void FeatureTracker::track_recover_feature(){

    std::vector<cv::Point2f> track_recover_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    // 跟踪恢复的图像
    cv::calcOpticalFlowPyrLK(cur_img, forw_img, recover_pts, track_recover_pts, status, err, cv::Size(21, 21), 3);

    for (int i = 0; i < int(forw_pts.size()); i++)
        if (status[i] && !inBorder(forw_pts[i]))
            status[i] = 0;

    reduceVector(track_recover_pts, status); // 删除跟踪失败的特征点
    reduceVector(recover_ids, status); // 删除跟踪失败的特征点
    reduceVector(recover_cnt, status); // 删除跟踪失败的特征点

    std::cout << "add size is " << track_recover_pts.size() << std::endl;

    for(int i = 0; i < track_recover_pts.size(); i++){

        cv::Point2f rec_pt = track_recover_pts[i];

        for(int j = 0; j < cur_pts.size(); j++){
            
            double dis = (Eigen::Vector2d(cur_pts[j].x, cur_pts[j].y) - Eigen::Vector2d(rec_pt.x, rec_pt.y)).norm(); // 两点之间的距离值

            if(dis < 0.8 * MIN_DIST){ // 距离足够近

                if(track_cnt[j] <= 2){ // 最新添加的特征点

                    // 替换
                    cur_pts[j] = rec_pt;
                    track_cnt[j] = ++recover_cnt[i];
                    ids[j] = recover_ids[i];

                    cv::Point2f pt_cam;
                    pt_img2cam(rec_pt, pt_cam);
                    cur_un_pts[j] = pt_cam;

                }
                else{

                    forw_pts.push_back(rec_pt);
                    track_cnt.push_back(++recover_cnt[i]);
                    ids.push_back(recover_ids[i]);
                    cv::Point2f pt_cam;
                    pt_img2cam(rec_pt, pt_cam);
                    cur_un_pts.push_back(pt_cam);
                }

                break; // 退出
            }
        } 
    }
}

void FeatureTracker::pt_img2cam(const cv::Point2f& pt_img, cv::Point2f& pt_cam){

    Eigen::Vector2d a(pt_img.x, pt_img.y);
    Eigen::Vector3d b;

    //根据不同的相机模型将二维坐标转换到三维坐标
    m_pinhole_camera->liftProjective(a, b);

    //再延伸到深度归一化平面上
    pt_cam = cv::Point2f(b.x() / b.z(), b.y() / b.z());
}