
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>


using namespace cv;
using namespace std;

double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;

	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;

	double cosine = (dx1*dx2 + dy1*dy2) / (sqrt(dx1*dx1 + dy1*dy1) * sqrt(dx2*dx2 + dy2*dy2) + 1e-10);

	return cosine;
}

enum string{ 红, 橙, 黄, 绿, 青, 蓝, 紫, 黑, 白, 其他 }colors;
void  get_point(vector<Point > contour,Point *p){
	if (contour.size() == 0)
		return ;
	int totalx = 0;
	int totaly = 0;
	int n = 0;
	for (size_t i = 0; i < contour.size(); i++){
		n++;
		totalx = totalx + contour[i].x;
		totaly = totaly + contour[i].y;
	}
	(*p).x = totalx / n;
	(*p).y = totaly / n;
}

int detect_color(Point p, Mat & src, Mat & rgb)
{
	
	int res=-1;
	int h=src.at<Vec3b>(p.y, p.x)[0];
	int s = src.at<Vec3b>(p.y, p.x)[1];
	int v = src.at<Vec3b>(p.y, p.x)[2];
	
	if (h >=0 && h <= 10 && s > 43  && v > 46 ){
		res = 0;
	}
	else if (h >= 156 && h <= 180 && s > 43 && v > 46){
		res = 0;
	}
	else if (h >= 11 && h <= 25 && s > 43 && v > 46){
		res = 1;
	}
	else if (h >= 26 && h <= 34 && s > 43 && v > 46){
		res = 2;
	}
	else if (h >= 35 && h <= 77 && s > 43 && v > 46){
		res = 3;
	}
	else if (h >= 78 && h <= 99 && s > 43 && v > 46){
		res = 4;
	}
	else if (h >= 100 && h <= 124 && s > 43 && v > 46){
		res = 5;
	}
	else if (h >= 125 && h <= 155 && s > 43 && v > 46){
		res = 6;
	}
	else 
	{
		int g = rgb.at<Vec3b>(p.y, p.x)[0];
		int b = rgb.at<Vec3b>(p.y, p.x)[1];
		int r = rgb.at<Vec3b>(p.y, p.x)[2];
		if (g <=10 && b <= 10 && r <=10){
			res = 7;
		}
		else if (g >=200 && b>=200 && r>=200){
			res = 8;
		}
		else{
			res = 9;
		}
	}
	return res;
}

void shape_rgb(Mat & src, Mat & rgb,CvSeq *first_contour, int *num_squares, vector<int>& squares_color, vector<vector<Point>>& squares, vector<vector<Point>>& circles, int *num_circles, vector<int>& circles_color, vector<vector<Point>>& triangles, int *num_triangles, vector<int>& triangles_color){
	vector<Point > contour;
	vector<Point > approx;
	Point poi = 0;
	int nu = -1;
	int cnt = 0;
	for (; first_contour != 0; first_contour = first_contour->h_next)
	{
		vector <Point>().swap(contour);
		vector <Point>().swap(approx);
		cnt++;
		for (int i = 0; i<first_contour->total; ++i) 
		{
			Point *p = (Point *)cvGetSeqElem(first_contour, i);
			Point pp = *p;
			contour.push_back(pp);
		}

		approxPolyDP(contour, approx, arcLength(contour, true)*0.02, true);
		squares.push_back(approx);
		
		if (approx.size() == 4 &&
			fabs(contourArea(approx)) >= 1000 &&
			isContourConvex(approx))
		{
			double maxCosine = 0;

			for (int j = 2; j < 5; j++)
			{
				double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
				maxCosine = MAX(maxCosine, cosine);
			}
			if (maxCosine <= 0.3)             //直角的矩形应该为0
			{
				squares.push_back(approx);
				*num_squares = *num_squares + 1;
				nu = -1;
				get_point(approx, &poi);
				int nu = detect_color(poi, src, rgb);
				squares_color.push_back(nu);

			}
			//re = re + 1;
		}
		else if (approx.size() == 3 &&fabs(contourArea(approx)) >= 500 &&isContourConvex(approx)){	//tt = tt + 1;	//
			triangles.push_back(approx);
			*num_triangles = *num_triangles + 1;
			nu = -1;
			get_point(approx, &poi);
			int nu = detect_color(poi, src, rgb);
			triangles_color.push_back(nu);
		}
		else if (approx.size() >=8 && fabs(contourArea(approx)) >= 500 && isContourConvex(approx)){	//tt = tt + 1;	//
			circles.push_back(approx);
			*num_circles = *num_circles + 1;
			nu = -1;
			get_point(approx, &poi);
			int nu = detect_color(poi, src, rgb);
			circles_color.push_back(nu);
		}
			}

}


void drawSquares(Mat img, vector<vector<Point>> squares)
{
	polylines(img, squares, true, Scalar(0, 0, 0), 1, LINE_AA);


}
int main()
{


	 IplImage *src = cvLoadImage("dect.jpg", 1);
	 IplImage* dst = cvCreateImage(cvGetSize(src), 8, 3);
	 IplImage* dst1 = cvCreateImage(cvGetSize(src), 8, 1);
	 IplImage* color_dst = cvCreateImage(cvGetSize(src), 8, 3);
	 CvMemStorage* storage = cvCreateMemStorage(0);
	 CvMemStorage* Rstorage = cvCreateMemStorage(0);
	 CvMemStorage* Gstorage = cvCreateMemStorage(0);
	 CvMemStorage* Bstorage = cvCreateMemStorage(0);
	 CvSeq* lines = 0;
	 int i;
	 IplImage* src1 = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);

	 IplImage*  RedImage = cvCreateImage(cvGetSize(src), 8, 1);
	 IplImage*  GreenImage = cvCreateImage(cvGetSize(src), 8, 1);
	 IplImage*  BlueImage = cvCreateImage(cvGetSize(src), 8, 1);

	 IplImage*  Rgray = cvCreateImage(cvGetSize(src), 8, 1);
	 IplImage*  Ggray = cvCreateImage(cvGetSize(src), 8, 1);
	 IplImage*  Bgray = cvCreateImage(cvGetSize(src), 8, 1);



	 cvSplit(src, BlueImage, GreenImage, RedImage, 0);

	 cvThreshold(BlueImage, Bgray, 100, 255, CV_THRESH_BINARY);
	 cvThreshold(GreenImage, Ggray, 100, 255, CV_THRESH_BINARY);
	 cvThreshold(RedImage, Rgray, 100, 255, CV_THRESH_BINARY);

	 vector<vector<Point>> contours;

	 cvZero(dst);
	 CvSeq *first_contour = NULL;
	 CvSeq *Rfirst_contour = NULL;
	 CvSeq *Gfirst_contour = NULL;
	 CvSeq *Bfirst_contour = NULL;
	

	 cvFindContours(Rgray, storage, &first_contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	 for (; first_contour != 0; first_contour = first_contour->h_next)
	 {
		 if (first_contour->h_next = 0)
			 break;
	
		 cvDrawContours(dst, first_contour, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 2, CV_FILLED, 8, cvPoint(0, 0));

			 }
	 cvCvtColor(dst, dst1, CV_BGR2GRAY);




	 cvFindContours(Rgray, Gstorage, &Gfirst_contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	 cvFindContours(Ggray, Rstorage, &Rfirst_contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	 cvFindContours(Bgray, Bstorage, &Bfirst_contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	 cvZero(Rgray);
	 cvZero(Bgray);
	 cvZero(Ggray);

	 cvDrawContours(Rgray, Rfirst_contour, 255, 255, 1, CV_FILLED, 8, cvPoint(0, 0));
	 cvDrawContours(Ggray, Gfirst_contour, 255, 255, 1, CV_FILLED, 8, cvPoint(0, 0));
	 cvDrawContours(Bgray, Bfirst_contour, 255,255, 1, CV_FILLED, 8, cvPoint(0, 0));
	
	 cvMerge(Bgray, Ggray, Rgray,0,dst);
	 cvCvtColor(dst, dst1, CV_BGR2GRAY);
	 cvThreshold(dst1, dst1, 240, 255, CV_THRESH_BINARY_INV);
	 cvFindContours(dst1, storage, &first_contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);


	 int num_squares=0;
	 vector<int> squares_color;
	 vector<vector<Point>> squares;
	 vector<vector<Point>> circles;
	 int num_circles=0; 
	 vector<int> circles_color;
	 vector<vector<Point>> triangles;
	 int num_triangles = 0;
	 vector<int> triangles_color;
	 Mat img;
	 img = cvarrToMat(src);
	 Mat imgHSV;
	
	 cvtColor(img, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV  
	 shape_rgb(imgHSV, img, first_contour, &num_squares, squares_color, squares, circles, &num_circles, circles_color, triangles, &num_triangles, triangles_color);

	 vector<int> mycircles_color;
	 IplImage* gray = cvCreateImage(cvGetSize(src), 8, 1);

	 CvMemStorage* mystorage = cvCreateMemStorage(0);
	 cvCvtColor(src, gray, CV_BGR2GRAY);
	 cvSmooth(gray, gray, CV_GAUSSIAN, 9, 9); // smooth it, otherwise a lot of false circles may be detected
	 CvSeq* mycircles = cvHoughCircles(gray, mystorage, CV_HOUGH_GRADIENT, 2, gray->height / 4, 200, 100);
	 Point circl = 0;
	 for (i = 0; i < mycircles->total; i++)
	 {
		 float* p = (float*)cvGetSeqElem(mycircles, i);
		 cvCircle(src, cvPoint(cvRound(p[0]), cvRound(p[1])), 3, CV_RGB(0, 255, 0), -1, 8, 0);
		 cvCircle(src, cvPoint(cvRound(p[0]), cvRound(p[1])), cvRound(p[2]), CV_RGB(255, 0, 0), 3, 8, 0);
		 circl.y = p[1];
		 circl.x = p[0];
		 int nu = detect_color(circl, imgHSV, img);
		 mycircles_color.push_back(nu);
	 }
	 drawSquares(img, squares);

	 printf("矩形数量：");
	 printf("%d", squares_color.size());
	 printf("\n");
	 printf("矩形颜色：");
	 printf("\n");
	 for (size_t i = 0; i < squares_color.size(); i++){
	
		 switch (squares_color[i])//判断枚举变量day的值  
		 {
		 case 0:printf("红\t"); break;
		 case 1:printf("橙\t"); break;
		 case 2:printf("黄\t"); break;
		 case 3:printf("绿\t"); break;
		 case 4:printf("青\t"); break;
		 case 5:printf("蓝\t"); break;
		 case 6:printf("紫\t"); break;
		 case 7:printf("黑\t"); break;
		 case 8:printf("白\t"); break;
		 case 9:printf("其他\t"); break;
		 }
		 //printf("\n");
	 }

	 printf("\n");
	 printf("圆形数量：");
	 printf("%d", num_circles);
	 printf("\n");
	 printf("圆形颜色：");
	 printf("\n");
	 for (size_t i = 0; i < circles_color.size(); i++){
	
		 switch (circles_color[i])//判断枚举变量day的值  
		 {
		 case 0:printf("红\t"); break;
		 case 1:printf("橙\t"); break;
		 case 2:printf("黄\t"); break;
		 case 3:printf("绿\t"); break;
		 case 4:printf("青\t"); break;
		 case 5:printf("蓝\t"); break;
		 case 6:printf("紫\t"); break;
		 case 7:printf("黑\t"); break;
		 case 8:printf("白\t"); break;
		 case 9:printf("其他\t"); break;
		 }
	 }
	 
	 printf("\n");
	 printf("三角形数量：");
	 printf("%d", num_triangles);
	 printf("\n");
	 printf("三角形颜色：");
	 printf("\n");
	 for (size_t i = 0; i < triangles_color.size(); i++){
		 switch (triangles_color[i])//判断枚举变量day的值  
		 {
		 case 0:printf("红\t"); break;
		 case 1:printf("橙\t"); break;
		 case 2:printf("黄\t"); break;
		 case 3:printf("绿\t"); break;
		 case 4:printf("青\t"); break;
		 case 5:printf("蓝\t"); break;
		 case 6:printf("紫\t"); break;
		 case 7:printf("黑\t"); break;
		 case 8:printf("白\t"); break;
		 case 9:printf("其他\t"); break;
		 }
	 }
	 printf("\n");





	 cvNamedWindow("原始图片", 1);
	cvShowImage("原始图片", dst);

	
	 cvNamedWindow("检测后图片", 1);
	 cvShowImage("检测后图片", src);
	 cvWaitKey(0);
	 cvDestroyWindow("原始图片");
	 cvDestroyWindow("检测后图片");
	
}

