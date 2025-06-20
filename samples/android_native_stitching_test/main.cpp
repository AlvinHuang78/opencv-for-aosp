#include <iostream>
#include <time.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

static inline int64_t getTimeMs()
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t) now.tv_sec*1000 + now.tv_nsec/1000000;
}

static inline int getTimeInterval(int64_t startTime)
{
    return int(getTimeMs() - startTime);
}

void warp(const cv::UMat& imgLeft, const cv::UMat& imgRight, const cv::Mat& H, cv::UMat& output) {
    // 1. Set the size of stitched image
    cv::Size canvasSize(imgLeft.cols + imgRight.cols, cv::max(imgLeft.rows, imgRight.rows));
    output = cv::UMat::zeros(canvasSize, CV_8UC3);

    // 2. Warp right image
    cv::UMat warpedRight;
    cv::warpPerspective(imgRight, warpedRight, H, canvasSize, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // 3. Draw left image on output
    imgLeft.copyTo(output(cv::Rect(0, 0, imgLeft.cols, imgLeft.rows)));

    // 4. Use the mask to draw warped right image on output
    cv::UMat gray, mask;
    cv::cvtColor(warpedRight, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY);
    warpedRight.copyTo(output, mask);  // 利用遮罩複製右圖的非黑像素

    // 5. 移除黑邊（使用 boundingRect）
    cv::UMat stitchedGray, stitchedMask;
    cv::cvtColor(output, stitchedGray, cv::COLOR_BGR2GRAY);
    cv::threshold(stitchedGray, stitchedMask, 1, 255, cv::THRESH_BINARY);
    cv::Rect roi = cv::boundingRect(stitchedMask);

    if (roi.area() > 0 && roi != cv::Rect(0, 0, output.cols, output.rows)) {
        output = output(roi);
    }
}

int main() {
    std::cout << "Testing OpenCV stitching performance." << std::endl;

    cv::Mat leftImg = cv::imread("/data/vendor/camera/taipei101_left.jpg");
    cv::Mat rightImg = cv::imread("/data/vendor/camera/taipei101_right.jpg");

    if (leftImg.empty() || rightImg.empty()) {
        std::cout << "Error reading images" << std::endl;
        return -1;
    } 
    std::cout << "Reads image done" << std::endl;

    int64_t t = getTimeMs();
    std::cout << "Start SIFT" << std::endl;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    cv::Mat descriptorsLeft, descriptorsRight;
    sift->detectAndCompute(leftImg, cv::noArray(), keypointsLeft, descriptorsLeft);
    sift->detectAndCompute(rightImg, cv::noArray(), keypointsRight, descriptorsRight);
    std::cout << "SIFT done:" << getTimeInterval(t) << "ms" << std::endl;

    std::cout << "Start BF Match" << std::endl;
    cv::Ptr<cv::BFMatcher> bfMatcher = cv::BFMatcher::create(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    bfMatcher->knnMatch(descriptorsLeft, descriptorsRight, knnMatches, 2);

    const float ratio = 0.75f;
    std::vector<cv::DMatch> goodMatches;
    for (auto& knnMatch : knnMatches) {
        if (knnMatch[0].distance < ratio * knnMatch[1].distance) {
            goodMatches.push_back(knnMatch[0]);
        }
    }

    std::vector<cv::Point2f> goodMatchPointsLeft, goodMatchPointsRight;
    for (auto& goodMatch : goodMatches) {
        goodMatchPointsLeft.push_back(keypointsLeft[goodMatch.queryIdx].pt);
        goodMatchPointsRight.push_back(keypointsRight[goodMatch.trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(goodMatchPointsRight, goodMatchPointsLeft, cv::RANSAC);

    cv::UMat uLeftImg, uRightImg;
    leftImg.copyTo(uLeftImg);
    rightImg.copyTo(uRightImg);

    t = getTimeMs();
    cv::UMat pano;
    warp(uLeftImg, uRightImg, H, pano);
    std::cout << "warp time:" << getTimeInterval(t) << "ms" << std::endl;

    //cv::imwrite("/data/vendor/camera/pano_result.jpg", pano);
    
    return 0;
}
