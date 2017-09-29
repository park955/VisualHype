#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

RNG rng(12345);

static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

Mat drawing(Mat image, vector<Point> vec, Point Target)
{
    //Draw the  square (vec) using polylines function
    const Point* p = &vec[0];
    int n = (int)vec.size();
    polylines(image, &p, &n, 1, true, Scalar(0,0,255), 3);
    
    //Draw the Crosshead using line function
    line(image, Point(Target.x-5,Target.y), Point(Target.x+5, Target.y), Scalar(255,0,0),3);
    line(image, Point(Target.x,Target.y-5), Point(Target.x, Target.y+5), Scalar(255,0,0),3);
    
    //Putting the X-coordinate of the crosshead at the top of the image
    //There IS better way of doing this, but I coded this sophomore year and it works.
    ostringstream cvt;
    string print;
    cvt << Target.x-(image.cols/2);
    print = cvt.str();
    putText(image, print, Point ((int)image.cols/2,40), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255,0,0),1);
    
    //Output the resultant image
    return image;
}

Point centerOfMass(vector<Point>& verticies)
{
    //Add up all the x and y coordinates of the polygon
    Point retVal = Point(0,0);
    for(int i=0; i<verticies.size(); i++)
    {
        retVal.x+=verticies[i].x;
        retVal.y+=verticies[i].y;
    }
    
    //Divide them = we are averaging the verticies = center of mass
    retVal.x/=verticies.size();
    retVal.y/=verticies.size();
    
    //Output the resultant Point
    return(retVal);
}


int main() {
    Mat source, blured, cannyMain, contoursImage;
    VideoCapture vidstream(0);
    namedWindow("CONTOURS");
    
    int screenIndex = 0;
    int colorIndex = 0;
    
    Scalar color[10];
    color[0] = Scalar(255,255,255);
    color[1] = Scalar(255,0,0);
    color[2] = Scalar(0,255,0);
    color[3] = Scalar(0,0,255);
    color[4] = Scalar(255,255,0);
    color[5] = Scalar(255,0,255);
    color[6] = Scalar(0,255,255);
    color[7] = Scalar(255,122,122);
    color[8] = Scalar(122,122,255);
    color[9] = Scalar(122,255,122);
    
    Point crossHead=Point(-1,1);
    
    while(1) {
        
        vidstream>>source;
        
        char c = waitKey(1);
        
        if(c == 27) break;
        else if(c>='0' && c<='9'){
            screenIndex = c-48;
        }
        
        else if(c == 32){
            int last = colorIndex;
            while(last == colorIndex){
                colorIndex = rng.uniform(0, 9);
            }
        }
        
        if( screenIndex == 1 ){
            
            Mat a;
            blur(source, a, Size(3, 3));
            Canny(a, cannyMain, 70, 140);
            
            vector<vector<Point>> contoursMain;
            vector<Vec4i> hierarchyMain;
            findContours(cannyMain, contoursMain, hierarchyMain, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
            
            contoursImage.create(source.rows, source.cols, source.type());
            contoursImage = Scalar(0, 0, 0);
            
            for(int i=0; i<contoursMain.size(); i++) {

                drawContours(contoursImage, contoursMain, i, color[colorIndex], 2, 8, hierarchyMain);
            
            }
            
            imshow("CONTOURS", contoursImage);
        }
        blur(source, blured, Size(3,3));
        cvtColor(blured, blured, CV_BGR2HSV);
        inRange(blured, Scalar(40,50,50), Scalar(98,255,255), blured);
            
        Mat mod,final=source.clone();
        Canny(blured, mod, 50, 100, 5);
      
        vector<vector<Point>> contours;
        vector<vector<Point>> goodones;
            
        findContours(mod,contours,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE);
            
        if(contours.size()){
                
            vector<Point> testing;
                
            for(int i=0; i<contours.size(); i++){

                approxPolyDP(Mat(contours[i]), testing, arcLength(Mat(contours[i])*0.045, true), true);
                    
                if(testing.size()==4 && fabs(contourArea(Mat(testing))) > 500 && isContourConvex(Mat(testing))){
                        
                    double maxCos = 0;
                    for(int j=2; j < 4+1; j++){
                        double cosine = fabs(angle(testing[j%4], testing[j-2], testing[j-1]));
                        maxCos = MAX(maxCos, cosine);
                    }
                    if(maxCos<.6){
                        goodones.push_back(testing);
                    }
                }
            }
        }
        //cout <<contours.size()<<" "<< goodones.size()<<endl;
        if (goodones.size()){
            crossHead = centerOfMass(goodones[0]);
            final = drawing(source, goodones[0], crossHead);
        }
            
        if(screenIndex==2) imshow("CONTOURS",blured);
        if(screenIndex==3) imshow("CONTOURS",mod);
        if(screenIndex==4) imshow("CONTOURS",final);
        if( screenIndex == 1 ){
            
            Mat a;
            blur(source, a, Size(3, 3));
            Canny(a, cannyMain, 70, 140);
            
            vector<vector<Point>> contoursMain;
            vector<Vec4i> hierarchyMain;
            findContours(cannyMain, contoursMain, hierarchyMain, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
            
            contoursImage.create(source.rows, source.cols, source.type());
            contoursImage = Scalar(0, 0, 0);
            
            if(crossHead.x>=0){
                for(int i=0; i<contoursMain.size(); i++) {
                    drawContours(contoursImage, contoursMain, i, Scalar(rng.uniform(0, 255),rng.uniform(0, 255),rng.uniform(0, 255)), 2, 8, hierarchyMain);
                }
            }
            else{
                for(int i=0; i<contoursMain.size(); i++) {
                    drawContours(contoursImage, contoursMain, i, color[colorIndex], 2, 8, hierarchyMain);
                }
            }
            imshow("CONTOURS", contoursImage);
        }
        
        else{
            imshow("CONTOURS",source);
        }
    }
}