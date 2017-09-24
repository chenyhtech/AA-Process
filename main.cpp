/*
	对于反走样文本识别
	AntiAliased Test
*/
#include</home/chen/SourceCode/opencv-test/include-main-modules.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include<sstream>
#include<cstdlib>
#include<set>
#include<list>
#include<list>
#include<queue>
#include<iterator>
#include<algorithm>
#include<utility>


//! ignore equivalent classes fewer than _CLASS_NUM_LOWER_BOUND_ 
#define _CLASS_NUM_LOWER_BOUND_ 6

//对于过大的bounding box的等价类，也要筛选去掉
#define _CLASS_BOUNDINGBOX_WIDTH 3000
#define _CLASS_BOUNDINGBOX_HEIGHT 400 

//Histogram test threshold
#define MAX_GROUP_NUM 150
#define MAX_BIN_NUM 550

using namespace std;
using namespace cv;


void gradientDetection( const Mat&, int, int,  list< set< pair<int , int> > >& );
//void gradientDetection( const Mat& ,Mat& ,Mat& , int , list< set< pair<int , int> > >& );

//找到每个等价类的矩形边界 (bounding boxes) 
//即每个等价类的左上角和右下角
void findRectRange( list< pair<int,int> >& , list< pair<int,int> >& , list< set< pair<int , int> > >& );

//在原始图像上画出等价类区域
void printRect( Mat&, list< pair<int,int> >& , list< pair<int,int> >&, int B,int G,int R);


//计算每个等价类区域对应的原始图像部分的histogram
//并且进行筛选
void calculateHist( Mat&, list< pair<int,int> >&, list< pair<int,int> >&, list< set< pair<int , int> > >& );


//颜色转变，直接在原始图像作用
void colorTransformation( Mat&, list< pair<int,int> >&, list< pair<int,int> >& );


// Character segmentation
void characterSegmentation( const Mat&, Mat&, int a[], list< pair<int,int> >&, list< pair<int,int> >& );



//------------Argument Explanation------------//
//2nd : pic name
//3rd : a constant for gradient-detection, a predetermined threshold ( 5 as default )
//4th : proximity threshold for gradient equivalence classes ( 2 as default )
//5th : character segmentation: threshold for init mask
//
//
//-------------End of Explanation-------------//

//------------  AA Text -------------//
int main( int argc , char** argv )
{
	//argc == 5, temp value,just for gradient detection
	// if( argc != 5 )
	// {
	// 	cout << "#----Command Format: ./test [pic_name] [gradient:threshold] [proximity] [character:threshold]" << endl;
	// 	return -1;
	// }

	Mat src = imread( argv[1], IMREAD_COLOR );
	Mat src2 = src.clone();		//histogram test
	Mat src3 = src.clone();		//color transformation
	Mat src4 = src.clone();     //character segmentation

	namedWindow("Source Picture" , WINDOW_AUTOSIZE);
	imshow("Source Picture" , src );

	Mat grayImg = imread( argv[1], IMREAD_GRAYSCALE );

	//---------------Index 6---------------//
	//......
	//---------------Index 6------------------//

	if( !grayImg.data  )
	{
		cout << "#----Image read error!!!!!!" << endl;
		return -1;
	}

	//!convert argv[2](a string) to an int value
	stringstream ss;
	int threshold_i ;
	string s( argv[2] );
	ss << s;
	ss >> threshold_i;

	stringstream ss2;
	int proximity;
	string x = argv[3];
	ss2 << x;
	ss2 >> proximity;
	cout << "#---- In process threshold: " << threshold_i << "  and proximity is: " << proximity << endl;
	//!convert

	//!Equivalent classes
	//所有的等价类
	list< set< pair<int , int> > > allEquivalentClass ;
	//等价类的左上角和右下角位置
	list< pair<int,int> > starts, ends ;

//--------------gradient detection------------//
	cout << "#----(main)Start gradient detection..." << endl;
	gradientDetection( grayImg, threshold_i, proximity, allEquivalentClass );
	cout << "#----(main):Gradient detection finished" << endl;
	
	findRectRange( starts, ends, allEquivalentClass );
	printRect( src2, starts, ends, 0, 0, 255);

	namedWindow("Before Histogram" , WINDOW_AUTOSIZE );
	imshow( "Before Histogram" , src2 );


//----------------Histogram------------------//
	cout << "#----(main):Start histogram test..." << endl;
	calculateHist( src, starts, ends, allEquivalentClass );
	cout << "#----(main):Calculate Histgram finished" << endl;

	findRectRange( starts, ends, allEquivalentClass );
	printRect( src, starts, ends, 0, 0, 255);
	
	namedWindow( "After histogram test" , WINDOW_AUTOSIZE );
	imshow("After histogram test" , src );


//----------------Color transformation-----------//	
	cout << "#----(main):Start color transformation..." << endl;
	colorTransformation( src3 , starts , ends );
	cout << "#----(main):Color transformation finished" << endl;
	
	namedWindow( "Color transformation" , WINDOW_AUTOSIZE );
	imshow("Color transformation" , src3 );


//-------------- Character segmentation ----------------//
	stringstream ss3;
	int threshold_1;
	string x1 = argv[4];
	ss3 << x1;
	ss3 >> threshold_1 ;

	int threshold_2,threshold_3,threshold_4,threshold_5,threshold_6;
	cout << "Local minima threshold:(100 120) " ; cin >> threshold_2;
	cout << "Local mixima : (180)" ; cin >> threshold_3;
	cout << "Stage1:maxima surrounding pixels threshold:(3 4)" ; cin >> threshold_4;
	cout << "Stage 2: gradient threshold:(200)" ; cin >> threshold_5;
	cout << "stage 3: brek points threshold:(1000)"; cin >> threshold_6; 
	cout << endl;

	int threshold[6] = { threshold_1 , threshold_2, threshold_3, threshold_4, threshold_5, threshold_6 };
	
	cout << "#----(main):Start character segmentation..." << endl;
	characterSegmentation( src3, src4, threshold, starts, ends );
	cout << "#----(main):Finish" << endl;

	namedWindow( "Character segmentation" , WINDOW_AUTOSIZE );
	imshow("Character segmentation" , src4 );
	
	imwrite( "/home/chen/桌面/after-segmentation.png", src4 );


	waitKey(0);
	return 0;

}


//!必须用于灰度(Gray Scale)图！
//--------------------------------对于 AA Text的梯度检测----------------------------// 
// img.depth() == 0 ; img.channels() == 1 ; img.isContinuous() == true;

void gradientDetection( const Mat& img, int delta, int proximity, list< set< pair<int , int> > >& allEquivalentClass )
{
	// accept only char type matrices
    CV_Assert(img.depth() == CV_8U);

	int nRows = img.rows , nCols = img.cols;	

	// ! index filter_index used to indicate Gradient index value
	//使用两个数组，其中一个在搜索等价类时会被更改，另一个单纯用于读，而不写;
	bool index[nRows][nCols] = {0};
	int filter_index[nRows][nCols] = {0} ;


	//-------- Gradient detection
	for( int i = 0 ; i < nRows ; i++ )
		for( int j = 0 ; j < nCols ; j++ )
		{	
			if( j != 0 && j != nCols -1 )
			{
				//-------针对 AA(反走样，反锯齿) 字体的梯度测试-----//
				if( img.ptr<uchar>(i)[j] > img.ptr<uchar>(i)[j-1] + delta && img.ptr<uchar>(i)[j] < img.ptr<uchar>(i)[j+1] - delta )
					index[i][j] = filter_index[i][j] = 1;

				else if( img.ptr<uchar>(i)[j] < img.ptr<uchar>(i)[j-1] - delta && img.ptr<uchar>(i)[j] > img.ptr<uchar>(i)[j+1] + delta )
				{
					index[i][j] = 1;
					filter_index[i][j] = -1;
				}
				else
					index[i][j] = filter_index[i][j] = 0;

				//-----------Index 5------//
				//只是梯度测试的另一种表示方法
				//-----------Index 5------//
			}
			else
				index[i][j] = filter_index[i][j] = 0;
		}

	//--------Find all equivalence classes in image 
	//!use <queue> to implement recursion!
	for( int i =0 ; i < nRows ; i++)
	{
		for( int j = 0 ; j < nCols ;j++)
		{
			set< pair<int,int> > temp;
			queue< pair<int,int> > q;
			
			if( index[i][j] != 0 )
				q.push( make_pair(i,j) );
			else
				continue;

			//! Create one equivalent class
			//! Change index[][] to complete finding process
			while( !q.empty() )
			{
				pair<int,int> current = q.front();
				q.pop();

				// 在该点周围 +-r , +-c的范围内
				for( int r = -2 ; r <= 2 ; r++ )
					for( int c = -proximity ; c <= proximity ; c++ )
					{
						if( r == 0 && c == 0 )
							continue;
						else
						{
							if( current.first+r>=0 && current.first+r<nRows && current.second+c>=0 && current.second+c<nCols 
									&& index[ current.first+r ][ current.second+c ] != 0 )
							{
								q.push( make_pair( current.first+r, current.second+c ) );
								index[ current.first+r ][ current.second+c ] = 0;
								temp.insert( make_pair( current.first+r, current.second+c ) );
							}
						}	
					}
				//--------------------//
			}

			allEquivalentClass.push_back( temp );
			//! Create one equivalent class
			
		}
	}


    //------------Index 1 -----------------//
	//！显示忽略之前的等价类数量
	cout << "#------(gradientDetection)Before ignoring: " << allEquivalentClass.size() << endl;
	//------------Index 1 -----------------//

	//---Display equivalence classes values-----//
	//-----------------Index 3-----------------//
	//-----------------Index 3-----------------//



	//--------- Ignore some equivalence classes -------------------//
	set< pair<int,int> > temp = allEquivalentClass.front();
	set< pair<int,int> >::iterator s_itr = temp.begin();

	bool type_positive, type_negative;

	int SIZE = allEquivalentClass.size();
	for( int i = 0 ; i < SIZE ; i++ )
	{
		temp = allEquivalentClass.front();
		allEquivalentClass.pop_front();

		// Not enough pixels
		if( temp.size() < _CLASS_NUM_LOWER_BOUND_ )
		{
			continue;
		}

		// Only one type of gradient 只有一种梯度的等价类要过滤掉
		s_itr = temp.begin();
		type_positive = false ;
	 	type_negative = false;

		for( ; s_itr != temp.end() ; ++ s_itr )
		{
			int x = s_itr->first , y = s_itr->second ;
			if( filter_index[x][y] == 1 )
				type_positive = true;
			else if( filter_index[x][y] == -1 )
				type_negative = true;
			else
				continue;
		}
		if( !(type_negative && type_positive) )
		{
			continue;
		}

		//Too large bounding boxes
		set< pair<int,int> >::iterator set_itr = temp.begin();
		int up, down, left, right;
		up = down = set_itr->first;
		left = right = set_itr->second;

		for( ; set_itr != temp.end() ; ++set_itr )
		{
			if( set_itr->first > down )
				down = set_itr->first;
			if( set_itr->first < up )
				up = set_itr->first;
			if( set_itr->second > right )
				right = set_itr->second;
			if( set_itr->second < left )
				left = set_itr->second;

		}
		int width = right - left;
		int height = down - up;
		if( width > _CLASS_BOUNDINGBOX_WIDTH || height > _CLASS_BOUNDINGBOX_HEIGHT )
			continue;


		// 如果满足条件，再添加到等价类集合中去
		allEquivalentClass.push_back( temp );
	}  
	
	//---------------Index 2----------------//
	//{...}
	//---------------Index 2----------------//

	cout << "#------(gradientDetection) After gradient detection: " << allEquivalentClass.size() << endl;


	//-----------------Index 3-----------------//
	//-----------------Index 3-----------------//
}


//找到每个等价类的矩形边界 (bounding boxes)  即每个等价类的左上角和右下角位置 
// starts是左上角， ends是右下角
//该函数应该起到更新作用，应该进行初始化(清空原有的starts , ends 的内容)
void findRectRange( list< pair<int,int> >& starts, list< pair<int,int> >& ends, list< set< pair<int , int> > >& allEquivalentClass )
{
	starts.clear();
	ends.clear();

	list< pair<int,int> > temp_starts, temp_ends;

	int up, down, left, right; //左上角， 右下角坐标

	set< pair<int,int> > temp = allEquivalentClass.front();
	set< pair<int,int> >::iterator set_itr = temp.begin();

	int SIZE = allEquivalentClass.size();
	for( int i = 0 ; i < SIZE ; i++ )
	{
		temp = allEquivalentClass.front();
		allEquivalentClass.pop_front();

		set_itr = temp.begin();

		up = down = set_itr->first;
		left = right = set_itr->second;

		for( ; set_itr != temp.end() ; ++set_itr )
		{
			if( set_itr->first > down )
				down = set_itr->first;
			if( set_itr->first < up )
				up = set_itr->first;
			if( set_itr->second > right )
				right = set_itr->second;
			if( set_itr->second < left )
				left = set_itr->second;

		}

		temp_starts.push_back( make_pair(up,left) );
		temp_ends.push_back( make_pair(down,right) );
		allEquivalentClass.push_back(temp);
	}
	
	starts = temp_starts;
	ends = temp_ends;

	//-------------Index 4--------------//
	//-------------Index 4--------------//
}

void printRect( Mat& img, list< pair<int,int> >& starts, list< pair<int,int> >& ends, int B, int G, int R)
{
	int SIZE = starts.size();
	assert( starts.size() == ends.size() );

	pair<int,int> dot_s , dot_e;

	for( int h = 0 ; h < SIZE ; h++ )
	{
		dot_s = starts.front();	dot_e = ends.front();
		starts.pop_front();	 ends.pop_front();
		// ! Point( col , row )
		rectangle( img, Point(dot_s.second, dot_s.first),
			Point(dot_e.second, dot_e.first),
			Scalar(B,G,R),
			1, LINE_8 );

		starts.push_back( dot_s );
		ends.push_back( dot_e );
	}
}

//image应该为原图（BGR图像）
void calculateHist( Mat& image, list< pair<int,int> >& starts, list< pair<int,int> >& ends, list< set< pair<int , int> > >& allEquivalentClass )
{
	int SIZE = starts.size();
	// make sure they are of same size
	assert( starts.size() == allEquivalentClass.size() && starts.size() == ends.size() );
	
	set< pair<int,int> >temp;	
	pair<int,int> dot_s , dot_e ;

	// 循环对每个区域进行分析
	Mat src;
	vector<Mat> bgr_planes;
	Mat b_hist, g_hist, r_hist;
	int histSize = 256;
	float range[] = {0,256};
	const float* histRange = { range };
  	bool uniform = true; bool accumulate = false;
 	
 	
 	int b_group_num = 0, g_group_num = 0, r_group_num = 0;
	int b_bin_num = 0, g_bin_num = 0, r_bin_num = 0;

	//For a bin group(in each channel) , bins_length > 1 or group size > 5( bin length == 1 )
	int group_length = 0 , group_size = 0;

	bool tooManyGroups, tooManyBins;

	for( int h = 0 ; h < SIZE ; h++ )
	{
		//pop first, if it is AA text,push_back later
		temp = allEquivalentClass.front();
		allEquivalentClass.pop_front();
		dot_s = starts.front();     dot_e = ends.front();
		starts.pop_front(); 		ends.pop_front();

		//找出原图对应的区域
		Rect rect( dot_s.second , dot_s.first, 
						( dot_e.second - dot_s.second + 1 ) ,( dot_e.first - dot_s.first + 1) ) ;
		image(rect).copyTo(src);

		
  		split( src, bgr_planes );

		calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
		calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
		calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );


		b_group_num = 0; g_group_num = 0; r_group_num = 0;
		b_bin_num = 0; g_bin_num = 0; r_bin_num = 0;

		// Set appropriate group_size threshold according to the area size
		int area_size = ( dot_e.second - dot_s.second ) * ( dot_e.first - dot_s.first );

		// Blue 
		for( int i = 0 ; i < 256 ; )
		{
			if( cvRound(b_hist.at<float>( i )) == 0 )
			{
				i++;
				continue;
			}
			group_size = 0 ;
			group_length = 0 ;
			while( cvRound(b_hist.at<float>( i )) && i < 256 )
			{
				group_length ++;
				group_size += cvRound(b_hist.at<float>( i )) ;
				i++;
			}
			if( group_length > 4 || group_size > 15 )
			{
				
				b_bin_num += group_length;
				b_group_num++;
			}

		}

		// Green Channel
		for( int i = 0 ; i < 256 ; )
		{
			if( cvRound(g_hist.at<float>( i )) == 0 )
			{
				i++;
				continue;
			}
			group_size = 0 ;
			group_length = 0 ;
			while( cvRound(g_hist.at<float>( i )) && i < 256 )
			{
				group_length ++;
				group_size += cvRound(g_hist.at<float>( i )) ;
				i++;
			}
			if( group_length > 4 || group_size > 15 )
			{
				
				g_bin_num += group_length;
				g_group_num++;
			}

		}

		// Red channel
		for( int i = 0 ; i < 256 ; )
		{
			if( cvRound(r_hist.at<float>( i )) == 0 )
			{
				i++;
				continue;
			}
			group_size = 0 ;
			group_length = 0 ;
			while( cvRound(r_hist.at<float>( i )) && i < 256 )
			{
				group_length ++;
				group_size += cvRound(r_hist.at<float>( i )) ;
				i++;
			}
			if( group_length > 4 || group_size > 15 )
			{
				r_bin_num += group_length;
				r_group_num++;
			}

		}

		tooManyGroups = ( b_group_num > MAX_GROUP_NUM || g_group_num > MAX_GROUP_NUM || r_group_num > MAX_GROUP_NUM )?true:false ;
		tooManyBins = ( b_bin_num > MAX_BIN_NUM || g_bin_num > MAX_BIN_NUM || r_bin_num > MAX_BIN_NUM )?true:false;

		if( tooManyBins || tooManyGroups )
			; // don't want this equvilance class
		else
		{
			allEquivalentClass.push_back(temp);
			starts.push_back(dot_s);
			ends.push_back(dot_e);
		}
	}

	//---Display equivalence class point values-----// 
//---------------------Index 3------------------//
	cout << "After Histogram: " << allEquivalentClass.size() << endl;
//-------------------End of Index 3 ------------------------//
}



void colorTransformation( Mat& image, list< pair<int,int> >& starts, list< pair<int,int> >& ends )
{
	int SIZE = starts.size();
	assert( starts.size() == ends.size() );

	pair<int,int> dot_s, dot_e;

	// 循环对每个区域进行分析
	// Mat src;
	// vector<Mat> bgr_planes;
	// Mat b_hist, g_hist, r_hist;
	int histSize = 256;
	float range[] = {0,256};
	const float* histRange = { range };
  	bool uniform = true; bool accumulate = false;
 	
	//For a bin group(in each channel) , bins_length > 1 or group size > 5( bin length == 1 )
	int group_length = 0 , group_size = 0;

	for( int h = 0 ; h < SIZE ; h++ )
	{
		dot_s = starts.front();     dot_e = ends.front();
		starts.pop_front(); 		ends.pop_front();

		Mat src;
		vector<Mat> bgr_planes;
		Mat b_hist, g_hist, r_hist;
		//找出原图对应的区域
		Rect rect( dot_s.second , dot_s.first, 
						( dot_e.second - dot_s.second + 1 ) ,( dot_e.first - dot_s.first + 1 ) ) ;
		image(rect).copyTo(src);

		
  		split( src, bgr_planes );

		calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
		calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
		calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );


	//----------index 1-------//
	//   Display three histogram pictures
	//-----------End----------//


		// Three tables, 
		int blue_table[256] = {0};
		int green_table[256] = {0};
		int red_table[256] = {0};
		for(int i = 0 ; i < 256; i++)
			blue_table[i] = green_table[i] = red_table[i] = 0;

		int index_start = 0, index_end = 0;

//------ blue,green,red table init ----------//

		// Blue channel analysis
		for( int i = 0 ; i < 256 ; )
		{
			if( cvRound(b_hist.at<float>( i )) == 0 )
			{
				i++;
				continue;
			}
			index_start = i;
			group_size = 0 ;	group_length = 0 ;
			while( cvRound(b_hist.at<float>( i )) && i < 256 )
			{
				group_length ++;
				group_size += cvRound(b_hist.at<float>( i )) ;
				i++;
			}
			index_end = i;
			if( group_size >= 4 )
				for( int i = index_start ; i < index_end ; i++ )
					blue_table[i] = cvRound(b_hist.at<float>( i ));

		}

		// Green channel
		index_start = 0; index_end = 0;
		for( int i = 0 ; i < 256 ; )
		{
			if( cvRound(g_hist.at<float>( i )) == 0 )
			{
				i++;
				continue;
			}
			index_start = i;
			group_size = 0 ;	group_length = 0 ;
			while( cvRound(g_hist.at<float>( i )) && i < 256 )
			{
				group_length ++;
				group_size += cvRound(g_hist.at<float>( i )) ;
				i++;
			}
			index_end = i;
			if( group_size >= 4 )
			{
				for( int i = index_start ; i < index_end ; i++ )
					green_table[i] = cvRound(g_hist.at<float>( i ));
			}

		}

		// Red
		index_start = 0; index_end = 0;
		for( int i = 0 ; i < 256 ; )
		{
			if( cvRound(r_hist.at<float>( i )) == 0 )
			{
				i++;
				continue;
			}
			index_start = i;
			group_size = 0 ;	group_length = 0 ;
			while( cvRound(r_hist.at<float>( i )) && i < 256 )
			{
				group_length ++;
				group_size += cvRound(r_hist.at<float>( i )) ;
				i++;
			}
			index_end = i;
			if( group_size >= 4 )
			{
				for( int i = index_start ; i < index_end ; i++ )
					red_table[i] =  cvRound(r_hist.at<float>( i ));
			}

		}



//-------------  Blue Green Red color analysis   -------------//

		//----------------Blue color analysis-----------//
		
		int blue_background_color = -1, blue_text_color = -1;
		int blue_background_color_tolerance = -1, blue_text_color_tolerance = -1;

		int bin_size = 1;
		for( ; bin_size <= 32 ; bin_size *= 2 )  // Decrease histogram resolution
		{
			//initialize a temp_table[] array
			int tempSize = 256 / bin_size;
			int temp_table[ tempSize ] = {0};
			for( int i = 0 ; i < tempSize ; i++ )
			{
				temp_table[i] = 0;
				for( int h = 0 ; h < bin_size ; h++ )
					temp_table[i] += blue_table[ i * bin_size + h ];
			}
			

			// Find exterior non zero bins( smallest and largest )
			// exterior bins range include left , include right
			int small_left = -1 , small_right = -1, large_left = -1, large_right = -1;
			for( int i = 0 ; i < tempSize ; )
			{
				if( temp_table[i] == 0 )
				{
					i++; 	continue;
				}
				large_left = i;
				if( small_left == -1 )
					small_left = i;
				while( temp_table[i] && i < tempSize )
					i++;
				
				large_right = i -1;
				if( small_right == -1 )
					small_right = i -1;
			}

			//Find the largest three sequential bins
			// index is the middle position
			int largest = -1;
			int index = -1;
			for( int i = 1 ; i < tempSize - 1 ; i++ )
			{
				if( temp_table[i-1] + temp_table[i] + temp_table[ i+1 ] > largest )
				{
					largest = temp_table[i-1] + temp_table[i] + temp_table[ i+1 ];
					index = i;
				}
			}
			if( index  == -1 )
				{	cerr<< "Process ERROR( color transformation ) " << endl; exit(0); }

			// Find the background index
			int background_index ;
			int max = index;
			if( temp_table[index - 1] > temp_table[max] )
				max = index - 1;
			if( temp_table[index + 1 ] > temp_table[max] )
				max = index + 1;

			background_index = max;

			// Check if the index is in one of the exterior bins
			int text_index = -1;
			if( background_index >= small_left && background_index <= small_right) 
			{
				text_index = (large_right + large_left ) / 2;
				blue_text_color_tolerance = ( (large_right-large_left)/2 > 0 )? ((large_right-large_left)/2) * bin_size : bin_size ;
				if( background_index - small_left > small_right - background_index )
					blue_background_color_tolerance = ( background_index - small_left )*bin_size ;
				else
					blue_background_color_tolerance = ( small_right - background_index )*bin_size ;
			}
			else if( background_index >= large_left && background_index <= large_right ) 
						
			{
				text_index = (small_left + small_right ) / 2;
				blue_text_color_tolerance = ( (small_right-small_left)/2 > 0 )? (small_right-small_left)/2 * bin_size : bin_size ;
				if( background_index - large_left > large_right - background_index )
					blue_background_color_tolerance = ( background_index - large_left )*bin_size ;
				else
					blue_background_color_tolerance = ( large_right - background_index )*bin_size ;
			}
			else
				continue;


			blue_background_color = ( background_index*bin_size + bin_size / 2 );
			blue_text_color = ( text_index*bin_size + bin_size / 2 );

			
			//---------- debug -----/-----------/
			// cout << "\n<Blue> bin size:" << bin_size << "   background index is:" << background_index 
			// 	<< "   text index is:" << text_index 
			// 	<< "  and the temp table contents:" << endl;
			// for( int i = 0 ; i < tempSize ; i++ )
			// 	cout << temp_table[i] << " ";
			// cout << endl;
			//-----//

			break;
		}



		//----------------Green color analysis----------//

		int green_background_color = -1, green_text_color = -1;
		int green_background_color_tolerance = -1, green_text_color_tolerance = -1;

		bin_size = 1;
		for( ; bin_size <= 32 ; bin_size *= 2 )  // Decrease histogram resolution
		{
			//initialize a temp_table[] array
			int tempSize = 256 / bin_size;
			int temp_table[ tempSize ] = {0};
			for( int i = 0 ; i < tempSize ; i++ )
			{
				temp_table[i] = 0;
				for( int h = 0 ; h < bin_size ; h++ )
					temp_table[i] += green_table[ i * bin_size + h ];
			}
			
			// Find exterior non zero bins( smallest and largest )
			// exterior bins range include left , include right
			int small_left = -1 , small_right = -1, large_left = -1, large_right = -1;
			for( int i = 0 ; i < tempSize ; )
			{
				if( temp_table[i] == 0 )
				{
					i++; 	continue;
				}
				large_left = i;
				if( small_left == -1 )
					small_left = i;
				while( temp_table[i] && i < tempSize )
				{
					i++;
				}
				large_right = i-1;
				if( small_right == -1 )
					small_right = i-1;
			}

			//Find the largest three sequential bins
			// index is the middle position
			int largest = -1;
			int index = -1;
			for( int i = 1 ; i < tempSize - 1 ; i++ )
			{
				if( temp_table[i-1] + temp_table[i] + temp_table[ i+1 ] > largest )
				{
					largest = temp_table[i-1] + temp_table[i] + temp_table[ i+1 ];
					index = i;
				}
			}
			if( index  == -1 )
				{	cerr<< "Process ERROR( color transformation ) " << endl; exit(0); }

			// Find the background index
			int background_index ;
			int max = index;
			if( temp_table[index - 1] > temp_table[max] )
				max = index - 1;
			if( temp_table[index + 1 ] > temp_table[max] )
				max = index + 1;

			background_index = max;

			// Check if the index is in one of the exterior bins
			int text_index = -1;
			if( background_index >= small_left && background_index <= small_right ) 
					
			{
				text_index = (large_right + large_left) / 2;
				green_text_color_tolerance = ( (large_right-large_left)/2 > 0 )? (large_right-large_left)/2 * bin_size : bin_size ;
				if( background_index - small_left > small_right - background_index )
					green_background_color_tolerance = ( background_index - small_left )*bin_size ;
				else
					green_background_color_tolerance = ( small_right - background_index )*bin_size ;
			}
			else if( background_index >= large_left && background_index <= large_right ) 
			{
				text_index = (small_left + small_right) / 2;
				green_text_color_tolerance = ( (small_right-small_left)/2 > 0 )? (small_right-small_left)/2 * bin_size : bin_size ;
				if( background_index - large_left > large_right - background_index )
					green_background_color_tolerance = ( background_index - large_left )*bin_size ;
				else
					green_background_color_tolerance = ( large_right - background_index )*bin_size ;
			}
			else
				continue;


			green_background_color = ( background_index * bin_size + bin_size / 2 );
			green_text_color = ( text_index * bin_size + bin_size / 2 );

			//---------- debug -----/-----------/
			// cout << "\n<Green> bin size:" << bin_size << "   background index is:" << background_index 
			// 	<< "   text index is:" << text_index 
			// 	<< "  and the temp table contents:" << endl;
			// for( int i = 0 ; i < tempSize ; i++ )
			// 	cout << temp_table[i] << " ";
			// cout << endl;
			//-----//

			break;
		}



		//---------------Red color analysis-------------//

		int red_background_color = -1, red_text_color = -1;
		int red_background_color_tolerance = -1, red_text_color_tolerance = -1;

		bin_size = 1;
		for( ; bin_size <= 32 ; bin_size *= 2 )  // Decrease histogram resolution
		{
			//initialize a temp_table[] array
			int tempSize = 256 / bin_size;
			int temp_table[ tempSize ] = {0};
			for( int i = 0 ; i < tempSize ; i++ )
			{
				temp_table[i] = 0;
				for( int h = 0 ; h < bin_size ; h++ )
					temp_table[i] += red_table[ i * bin_size + h ];
			}
			
			// Find exterior non zero bins( smallest and largest )
			// exterior bins range include left , include right
			int small_left = -1 , small_right = -1, large_left = -1, large_right = -1;
			for( int i = 0 ; i < tempSize ; )
			{
				if( temp_table[i] == 0 )
				{
					i++; 	continue;
				}
				large_left = i;
				if( small_left == -1 )
					small_left = i;
				while( temp_table[i] && i < tempSize )
				{
					i++;
				}
				large_right = i-1;
				if( small_right == -1 )
					small_right = i-1;
			}

			//Find the largest three sequential bins
			// index is the middle position
			int largest = 0;
			int index = -1;
			for( int i = 1 ; i < tempSize - 1 ; i++ )
			{
				if( temp_table[i-1] + temp_table[i] + temp_table[ i+1 ] > largest )
				{
					largest = temp_table[i-1] + temp_table[i] + temp_table[ i+1 ];
					index = i;
				}
			}
			if( index  == -1 )
				{	cerr<< "Process ERROR( color transformation ) " << endl; exit(0); }

			// Find the background index
			int background_index ;
			int max = index;
			if( temp_table[index - 1] > temp_table[max] )
				max = index - 1;
			if( temp_table[index + 1 ] > temp_table[max] )
				max = index + 1;
			background_index = max;

			// Check if the index is in one of the exterior bins
			int text_index = -1;
			if( background_index >= small_left && background_index <= small_right ) 
			{
				text_index = (large_right + large_left) / 2;
				red_text_color_tolerance = ( (large_right-large_left)/2 > 0 )? (large_right-large_left)/2 * bin_size : bin_size ;
				if( background_index - small_left > small_right - background_index )
					red_background_color_tolerance = ( background_index - small_left )*bin_size ;
				else
					red_background_color_tolerance = ( small_right - background_index )*bin_size ;
			}
			else if( background_index >= large_left && background_index <= large_right) 
			{
				text_index = (small_left + small_right) / 2;
				red_text_color_tolerance = ( (small_right-small_left)/2 > 0 )? (small_right-small_left)/2 * bin_size : bin_size ;
				if( background_index - large_left > large_right - background_index )
					red_background_color_tolerance = ( background_index - large_left )*bin_size ;
				else
					red_background_color_tolerance = ( large_right - background_index )*bin_size ;
			}
			else
				continue;


			red_background_color = ( background_index * bin_size + bin_size / 2 );
			red_text_color = ( text_index * bin_size + bin_size / 2 );
			//!!!Attention!!!Not sure how to define it
			//red_text_color_tolerance = bin_size ;


			//---------- debug -----/-----------/
			// cout << "\n<Red> bin size:" << bin_size << "   background index is:" << background_index 
			// 	<< "   text index is:" << text_index 
			// 	<< "  and the temp table contents:" << endl;
			// for( int i = 0 ; i < tempSize ; i++ )
			// 	cout << temp_table[i] << " ";
			// cout << endl;
			//-----//


			break;
		}



//----------------(debug), Show origin image, and compare with calculated color ------------//

		// cout << "\nBackground color----->  B:" << blue_background_color << "  G:" << green_background_color << " R:" << red_background_color
		// 	<< "\nText color----->   B:" << blue_text_color << "  G:" << green_text_color << "  R:" << red_text_color << endl << endl;
		
		// namedWindow("Origin Image" , WINDOW_AUTOSIZE );
		// imshow( "Origin Image" , src );

		// Mat background( 100 , 100 , CV_8UC3, Scalar( blue_background_color , green_background_color , red_background_color) );
		// Mat text(100, 100 , CV_8UC3, Scalar(blue_text_color, green_text_color, red_text_color) );

		// namedWindow("Background" , WINDOW_AUTOSIZE );
		// imshow("Background" , background );
		// namedWindow("Text" , WINDOW_AUTOSIZE );
		// imshow("Text" , text);
		// waitKey(0);
//-----------//




//------------------  Decoloring  --------------------//
		int b_back_left = ((blue_background_color - blue_background_color_tolerance) >= 0)?(blue_background_color-blue_background_color_tolerance) : 0 ;
		int b_back_right=((blue_background_color + blue_background_color_tolerance) <= 255)?(blue_background_color+blue_background_color_tolerance):255;
		int b_text_left = ( (blue_text_color - blue_text_color_tolerance) >= 0)?(blue_text_color-blue_text_color_tolerance): 0;
		int b_text_right= ( (blue_text_color + blue_text_color_tolerance) <= 255)?(blue_text_color + blue_text_color_tolerance):255;

		int small = b_back_left, large = b_back_right;
		if( b_text_left < small )
			small = b_text_left;
		if( b_text_right > large )
			large = b_text_right;

		int blue_identical ;
		if( (large - small) <= (b_back_right-b_back_left)+(b_text_right-b_text_left) )
			blue_identical = 1;
		else
			blue_identical = 0;

		int g_back_left = ((green_background_color - green_background_color_tolerance) >= 0)?(green_background_color - green_background_color_tolerance):0 ;
		int g_back_right= ((green_background_color + green_background_color_tolerance) <= 255)?(green_background_color + green_background_color_tolerance):255;
		int g_text_left = ((green_text_color - green_text_color_tolerance) >= 0)?(green_text_color - green_text_color_tolerance): 0;
		int g_text_right= ((green_text_color + green_text_color_tolerance) <= 255)?(green_text_color + green_text_color_tolerance):255;

		small = g_back_left; large = g_back_right;
		if( g_text_left < small )
			small = g_text_left;
		if( g_text_right > large )
			large = g_text_right;

		int green_identical ;
		if( (large - small) <= (g_back_right-g_back_left)+(g_text_right-g_text_left) )
			green_identical = 1;
		else
			green_identical = 0;

		int r_back_left = ((red_background_color - red_background_color_tolerance) >= 0)?(red_background_color - red_background_color_tolerance):0;
		int r_back_right= ((red_background_color + red_background_color_tolerance) <= 255)?(red_background_color + red_background_color_tolerance):255;
		int r_text_left = ((red_text_color - red_text_color_tolerance) >= 0)?(red_text_color - red_text_color_tolerance): 0;
		int r_text_right= ((red_text_color + red_text_color_tolerance) <= 255)?(red_text_color + red_text_color_tolerance):255;

		small = r_back_left; large = r_back_right;
		if( r_text_left < small )
			small = r_text_left;
		if( r_text_right > large )
			large = r_text_right;

		int red_identical ;
		if( (large - small) <= (r_back_right-r_back_left)+(r_text_right-r_text_left) )
			red_identical = 1;
		else
			red_identical = 0;


		int identical_channels = blue_identical + green_identical + red_identical;

		//--- Index whether a sub-pixel need to be fixed ---//
		bool needFix[ dot_e.first - dot_s.first +1 ][ 3*(dot_e.second - dot_s.second +1) ] = {false};

		for( int r = 0 ; r < dot_e.first - dot_s.first +1 ; r++ )
		{
			for( int c = 0 ; c < (dot_e.second - dot_s.second +1) ; c++ )
				needFix[r][3*c] = needFix[r][3*c + 1] = needFix[r][3*c+2] = false;
			
		}


//-------- debug; show:1,ientical channels;  2,needFix[][] values-----//
		// cout << "\nBlue channel:" << blue_identical << "  Green:" << green_identical << "  Red:" << red_identical << endl;
		// cout << "Identical channel number:" << identical_channels << endl;

		// //three channels back and text range
		// cout << "B background:" << b_back_left << "~" << b_back_right << " Text:" << b_text_left << "~" << b_text_right << endl;
		// cout << "G background:" << g_back_left << "~" << g_back_right << " Text:" << g_text_left << "~" << g_text_right << endl;
		// cout << "R background:" << r_back_left << "~" << r_back_right << " Text:" << r_text_left << "~" << r_text_right << endl;


		// cout << "Init value:" << endl;
		// for( int r = 0 ; r < dot_e.first - dot_s.first +1 ; r++ )
		// {
		// 	for( int c = 0 ; c < (dot_e.second - dot_s.second +1) ; c++ )
		// 	{
		// 		cout << needFix[r][3*c] << " " << needFix[r][3*c + 1] << " " << needFix[r][3*c+2] << ";  ";
		// 	}
		// 	cout << endl;
		// }


//----------------- transform ------------------//		
		for( int row = dot_s.first ; row <= dot_e.first ; row++ )
			for( int col = dot_s.second; col <= dot_e.second; col++ )
			{
				// Blue
				int sub_blue = image.ptr<uchar>(row)[ 3*col + 0 ];
				if( blue_identical )
					needFix[ row-dot_s.first ][ 3*(col-dot_s.second) ] = true;
				else
				{
					if( sub_blue >= b_back_left && sub_blue <= b_back_right )
						image.ptr<uchar>(row)[ 3*col + 0 ] = 255;
					else if( sub_blue >= b_text_left && sub_blue <= b_text_right )
						image.ptr<uchar>(row)[ 3*col + 0 ] = 0 ;
					else
						image.ptr<uchar>(row)[ 3*col + 0 ] = 255 * (sub_blue - blue_text_color) / (blue_background_color - blue_text_color);
				}
				
				
				// Green
				int sub_green = image.ptr<uchar>(row)[ 3*col + 1 ];
				if( green_identical )
					needFix[ row-dot_s.first ][ 3*(col-dot_s.second) +1] = 1;
				else
				{
					if( sub_green >= g_back_left && sub_green <= g_back_right )
						image.ptr<uchar>(row)[ 3*col + 1 ] = 255;
					else if( sub_green >= g_text_left && sub_green <= g_text_right )
						image.ptr<uchar>(row)[ 3*col + 1 ] = 0 ;
					else	
						image.ptr<uchar>(row)[ 3*col + 1 ] = 255 * (sub_green - green_text_color) / (green_background_color - green_text_color);		
				}
				

				// Red
				int sub_red = image.ptr<uchar>(row)[ 3*col + 2 ];
				if( red_identical )
					needFix[ row-dot_s.first ][ 3*(col-dot_s.second) +2 ] = 1;
				else
				{
					if( sub_red >= r_back_left && sub_red <= r_back_right )
						image.ptr<uchar>(row)[ 3*col + 2 ] = 255;
					else if( sub_red >= r_text_left && sub_red <= r_text_right )
						image.ptr<uchar>(row)[ 3*col + 2 ] = 0 ;
					else
						image.ptr<uchar>(row)[ 3*col + 2 ] = 255 * (sub_red - red_text_color) / (red_background_color - red_text_color);
				}
				
			}


//-------- debug; show needFix[][] values-----//
		// cout << "After process" << endl;
		// for( int r = 0 ; r < dot_e.first - dot_s.first +1 ; r++ )
		// {
		// 	for( int c = 0 ; c < (dot_e.second - dot_s.second +1) ; c++ )
		// 	{
		// 		cout << needFix[r][3*c] << " " << needFix[r][3*c + 1] << " " << needFix[r][3*c+2] << ";  ";
		// 	}
		// 	cout << endl;
		// }


//-------------- When one or two channels are same, process relevant sub-pixels
		for( int r = 0 ; r < dot_e.first - dot_s.first +1 ; r++ )
			for( int c = 0 ; c < (dot_e.second - dot_s.second +1) ; c++ )
			{
				//blue
				if( needFix[r][ 3*c ] )
				{
					if( identical_channels == 1 )
					{
						image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) ] = 
						( image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) + 1 ] 
							+ image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) + 2 ]  ) / 2 ;
					}
					else  // two identical channels
					{
						if( needFix[r][ 3*c + 1 ] ) // green is also identical
							image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) ] =
								image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) + 2 ] ;
						else // red is identical
							image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) ] =
								image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) + 1 ] ;
					}
				}

				// green
				if( needFix[r][ 3*c +1 ] )
				{
					if( identical_channels == 1 )
					{
						image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) +1 ] = 
						( image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) ] 
							+ image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) + 2 ]  ) / 2 ;
					}
					else // two identical channels
					{
						if( needFix[r][ 3*c ] ) // blue is also identical
							image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) +1 ] =
								image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) + 2 ] ;
						else // red is identical
							image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) +1 ] =
								image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) ] ;
					}
				}

				//red
				if( needFix[r][ 3*c +2 ] )
				{
					if( identical_channels == 1 )
					{
						image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) +2 ] = 
						( image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) ] 
							+ image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) + 1 ]  ) / 2 ;
					}
					else // two identical channels
					{
						if( needFix[r][ 3*c ] ) // blue is also identical
							image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) +2 ] =
								image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) + 1 ] ;
						else // green is identical
							image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) +2 ] =
								image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) ] ;
					}
				}
			}
//----//


// //make one pixel's 3 sub-pixels have identical value
		for( int r = 0 ; r < dot_e.first - dot_s.first +1 ; r++ )
			for( int c = 0 ; c < (dot_e.second - dot_s.second +1) ; c++ )
			{
				int result = image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) ] 
							+ image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) +1 ]
							+ image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) +2 ];
				result = result / 3;
				
				image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) ] = result;
				image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) +1] = result;
				image.ptr<uchar>( dot_s.first + r )[ 3*(dot_s.second + c) +2] = result;
			}
//---//

		starts.push_back( dot_s );
		ends.push_back( dot_e );
	}

}



// Mat is the image after colorTransformation( image = colorTransformation's image )
// outImage is the final mask( binary: black on white )  CV_8UC1

void characterSegmentation( const Mat& image, Mat& outImage, int threshold[6], list< pair<int,int> >& starts, list< pair<int,int> >& ends )
{
	/*
	parameter list

	Stage1: threshold_2(100), threshold_3(160-180), threshold_4(2 3 4)
	Stage2: threshold_5(180)
	Stage3: threshold_6(1000)
	*/

	int imageRows = outImage.rows;
	int imageCols = outImage.cols;
	for( int r = 0 ; r < imageRows; r++ )
		for( int c = 0 ; c < imageCols ; c++ )
		{
			outImage.ptr<uchar>(r)[3*c + 0] = 255;
			outImage.ptr<uchar>(r)[3*c + 1] = 255;
			outImage.ptr<uchar>(r)[3*c + 2] = 255;
		}

	int SIZE = starts.size();
	pair<int,int> dot_s, dot_e;

	for( int h = 0 ; h < SIZE ; h++ )
	{
		dot_s = starts.front();  starts.pop_front();
		dot_e = ends.front();    ends.pop_front();

		int nRows = dot_e.first - dot_s.first + 1;
		int nCols = dot_e.second - dot_s.second + 1;

		bool mask[nRows][nCols] = {false};
		bool breakpoint[nRows][nCols] = {false};		
		for( int r = 0 ; r < nRows ; r++ )
			for( int c = 0 ; c < nCols ; c++ )
				mask[r][c] = breakpoint[r][c] = false;


	// Stage 1
	int threshold_2 = threshold[1]; 
	int threshold_3 = threshold[2]; 
	int threshold_4 = threshold[3]; //3,4...;
		//1: below a threshold
		for( int r = 0 ; r < nRows ; r++ )
			for( int c = 0 ; c < nCols ; c++ )
				if( image.ptr<uchar>(dot_s.first + r)[ 3*(dot_s.second + c) ] < threshold[0] )
					mask[r][c] = true;

		//-----debug: how many pixels are inside the mask----//
		int maskSize = 0;
		for( int r = 0 ; r < nRows ; r++ )
			for( int c = 0 ; c < nCols ; c++ )
				if( mask[r][c] )
					maskSize++;
		//cout << "Original Mask size:" << maskSize << " ";

		//2: local minima and maxima
		// Point minLocation, maxLocation;
		// double minVal, maxVal;
		
		// int length = 3;
		// for( int row = dot_s.first ; row <= dot_e.first - ( length-2 ) ; row++ )
		// 	for( int col = dot_s.second ; col <= dot_e.second - (length-1-1) ; col++ )
		// 	{
		// 		Mat temp;
		// 		Rect rect( col, row, length-1, length-1 );  //(x,y,width,height)
		// 		image(rect).copyTo( temp );
		// 		vector<Mat> bgr_planes;
		// 		split( temp, bgr_planes );
		// 		Mat proc = bgr_planes[0];

		// 		minMaxLoc( proc, &minVal, &maxVal, &minLocation, &maxLocation );

		// 		if( minVal < threshold_2 )
		// 			mask[ row + minLocation.y - dot_s.first ][ col + minLocation.x - dot_s.second ] = true;

		// 		if( maxVal > threshold_3 )
		// 			;
		// 		else
		// 		{
		// 			mask[ row + maxLocation.y - dot_s.first ][ col + maxLocation.x - dot_s.second ] = true;
		// 			int surroundingSum = 0;
		// 			for( int r = -1 ; r <=1 ; r++ )
		// 				for( int c = -1; c <=1 ; c++ )
		// 				{
		// 					if( r == 0 && c == 0 )
		// 						continue;
		// 					surroundingSum += (mask[row+maxLocation.y+r-dot_s.first ][col+maxLocation.x + c -dot_s.second])? 1:0 ;
		// 				}

		// 			if( surroundingSum < threshold_4 )
		// 				breakpoint[ row + maxLocation.y - dot_s.first ][ col + maxLocation.x - dot_s.second ] = true;
				
		// 		}
		// 	}

		//-----debug: how many pixels are inside the mask----//
		maskSize = 0;
		for( int r = 0 ; r < nRows ; r++ )
			for( int c = 0 ; c < nCols ; c++ )
				if( mask[r][c] )
					maskSize++ ;
		//cout << "  After Stage1:" << maskSize ;
	//--------//

	

	// Stage2
	// int threshold_5 = threshold[4]; //200
	// 	bool condition_1 =false, condition_2 = false, condition_3 = false;
	// 	// 1: left to right
	// 	for( int row = dot_s.first ; row <= dot_e.first ; row++ )
	// 		for( int col = dot_s.second + 1 ; col <= dot_e.second -1 ; col++ )
	// 		{
	// 			int pre_col = col -1;
	// 			int next_col = col +1;
	// 			if( image.ptr<uchar>(row)[3*col] < threshold_5 )
	// 				condition_1 = true;
	// 			else
	// 				continue;
	// 			if( (mask[row-dot_s.first][pre_col-dot_s.second] && mask[row-dot_s.first][next_col-dot_s.second] == false) 
	// 				|| (mask[row-dot_s.first][pre_col-dot_s.second] == false && mask[row-dot_s.first][next_col-dot_s.second] ) )
	// 				condition_2 == true;
	// 			else
	// 				continue;
				
	// 			if( mask[row-dot_s.first][pre_col-dot_s.second] ) //left pixel is in mask
	// 			{
	// 				if( image.ptr<uchar>(row)[3*col] >= image.ptr<uchar>(row)[3*pre_col] 
	// 					&& image.ptr<uchar>(row)[3*col] < image.ptr<uchar>(row)[3*next_col] )
	// 					condition_3 == true;
	// 				else
	// 					continue;
	// 			}
	// 			else
	// 			{
	// 				if( image.ptr<uchar>(row)[3*col] >= image.ptr<uchar>(row)[3*next_col] 
	// 					&& image.ptr<uchar>(row)[3*col] < image.ptr<uchar>(row)[3*pre_col] )
	// 					condition_3 == true;
	// 				else
	// 					continue;
	// 			}
	// 			mask[ row - dot_s.first ][ col - dot_s.second ] = true;
	// 		}
		
	// 	condition_1 = condition_2 = condition_3 = false;
		
	// 	// right to left
	// 	for( int row = dot_s.first ; row <= dot_e.first ; row++ )
	// 		for( int col = dot_e.second - 1 ; col >= dot_s.second +1 ; col-- )
	// 		{
	// 			int pre_col = col -1;
	// 			int next_col = col +1;
	// 			if( image.ptr<uchar>(row)[3*col] < threshold_5 )
	// 				condition_1 = true;
	// 			else
	// 				continue;
	// 			if( (mask[row-dot_s.first][pre_col-dot_s.second] && mask[row-dot_s.first][next_col-dot_s.second] == false) 
	// 				|| (mask[row-dot_s.first][pre_col-dot_s.second] == false && mask[row-dot_s.first][next_col-dot_s.second] ) )
	// 				condition_2 == true;
	// 			else
	// 				continue;
				
	// 			if( mask[row-dot_s.first][pre_col-dot_s.second] ) //left pixel is in mask
	// 			{
	// 				if( image.ptr<uchar>(row)[3*col] >= image.ptr<uchar>(row)[3*pre_col] 
	// 					&& image.ptr<uchar>(row)[3*col] < image.ptr<uchar>(row)[3*next_col] )
	// 					condition_3 == true;
	// 				else
	// 					continue;
	// 			}
	// 			else
	// 			{
	// 				if( image.ptr<uchar>(row)[3*col] >= image.ptr<uchar>(row)[3*next_col] 
	// 					&& image.ptr<uchar>(row)[3*col] < image.ptr<uchar>(row)[3*pre_col] )
	// 					condition_3 == true;
	// 				else
	// 					continue;
	// 			}
	// 			mask[ row - dot_s.first ][ col - dot_s.second ] = true;
	// 		}

	// 	//-----debug: how many pixels are inside the mask----//
	// 	maskSize = 0;
	// 	for( int r = 0 ; r < nRows ; r++ )
	// 		for( int c = 0 ; c < nCols ; c++ )
	// 			if( mask[r][c] )
	// 				maskSize++ ;
	// 	//cout << "  After Stage2:" << maskSize << endl;
	// //-------//




	// // Stage 3   Find potential breakpoints
	int threshold_6 = threshold[5]; //1000;
	// 3.1:
    	// for( int row = dot_s.first +1 ; row <= dot_e.first -2 ; row++ )
    	// 	for( int col = dot_s.second +1 ; col <= dot_e.second -2 ; col++ )
    	// 	{
    	// 		int surroundingSum = 0;

    	// 		if( mask[ row-dot_s.first ][ col-dot_s.second] && mask[ row-dot_s.first+1 ][ col-dot_s.second+1 ]
    	// 			&& mask[ row-dot_s.first+1 ][ col-dot_s.second]==false 
    	// 			&& mask[ row-dot_s.first ][ col-dot_s.second +1] == false )
    	// 		{
    	// 			surroundingSum = image.ptr<uchar>(row-1)[3*(col-1)] + image.ptr<uchar>(row-1)[3*(col)]
    	// 							+ image.ptr<uchar>(row)[3*(col-1)] + image.ptr<uchar>(row+1)[3*(col+2)]
    	// 							+ image.ptr<uchar>(row+2)[3*(col+1)] + image.ptr<uchar>(row+2)[3*(col+2)];
    	// 			if( surroundingSum > threshold_6 )
    	// 			{
    	// 				breakpoint[ row-dot_s.first ][ col-dot_s.second ] = true;
    	// 				breakpoint[ row-dot_s.first+1 ][ col-dot_s.second+1 ] = true;
    	// 			}
    	// 		}
    	// 		else if( ! mask[ row-dot_s.first ][ col-dot_s.second] && ! mask[ row-dot_s.first+1 ][ col-dot_s.second+1 ]
    	// 				&& mask[ row-dot_s.first+1 ][ col-dot_s.second] 
    	// 				&& mask[ row-dot_s.first ][ col-dot_s.second +1] )
    	// 		{
    	// 			surroundingSum =  image.ptr<uchar>(row+1)[3*(col-1)] + image.ptr<uchar>(row+2)[3*(col-1)]
    	// 							+ image.ptr<uchar>(row+2)[3*(col)] + image.ptr<uchar>(row-1)[3*(col+1)]
    	// 							+ image.ptr<uchar>(row-1)[3*(col+2)] + image.ptr<uchar>(row)[3*(col+2)];

    	// 			if( surroundingSum > threshold_6 )
    	// 			{
    	// 				breakpoint[ row-dot_s.first+1 ][ col-dot_s.second ] = true;
    	// 				breakpoint[ row-dot_s.first ][ col-dot_s.second+1 ] = true;
    	// 			}
    	// 		}
    	// 		else
    	// 			continue;
    	// 	}

    // 3.2:
    	// for( int row = dot_s.first +1 ; row <= dot_e.first -2 ; row++ )
    	// 	for( int col = dot_s.second +1 ; col <= dot_e.second -3 ; col++ )
    	// 	{
    	// 		int surroundingSum = 0;
    	// 		if( !mask[row-dot_s.first][col-dot_s.second] && mask[row-dot_s.first][col-dot_s.second+1] 
    	// 			&& mask[row-dot_s.first][col-dot_s.second+2] && mask[row-dot_s.first+1][col-dot_s.second]
    	// 			&& mask[row-dot_s.first+1][col-dot_s.second+1] && !mask[row-dot_s.first+1][col-dot_s.second+2] )
    	// 		{
    	// 			surroundingSum = image.ptr<uchar>(row-1)[3*(col)] + image.ptr<uchar>(row-1)[3*(col+1)]
    	// 							+image.ptr<uchar>(row-1)[3*(col+2)] + image.ptr<uchar>(row+2)[3*(col)]
    	// 							+image.ptr<uchar>(row+2)[3*(col+1)] + image.ptr<uchar>(row+2)[3*(col+2)];

    	// 			if( surroundingSum > threshold_6 )
    	// 			{
    	// 				breakpoint[ row-dot_s.first ][ col-dot_s.second +1] =true;
    	// 				breakpoint[ row-dot_s.first +1][ col-dot_s.second +1] =true;
    	// 			}
    	// 		}
    	// 		else if( mask[row-dot_s.first][col-dot_s.second] && mask[row-dot_s.first][col-dot_s.second+1] 
    	// 			&& !mask[row-dot_s.first][col-dot_s.second+2] && !mask[row-dot_s.first+1][col-dot_s.second]
    	// 			&& mask[row-dot_s.first+1][col-dot_s.second+1] && !mask[row-dot_s.first+1][col-dot_s.second+2] )
    	// 		{
    	// 			surroundingSum = image.ptr<uchar>(row-1)[3*(col)] + image.ptr<uchar>(row-1)[3*(col+1)]
    	// 							+image.ptr<uchar>(row-1)[3*(col+2)] + image.ptr<uchar>(row+2)[3*(col)]
    	// 							+image.ptr<uchar>(row+2)[3*(col+1)] + image.ptr<uchar>(row+2)[3*(col+2)];

    	// 			if( surroundingSum > threshold_6 )
    	// 			{
    	// 				breakpoint[ row-dot_s.first ][ col-dot_s.second +1] =true;
    	// 				breakpoint[ row-dot_s.first +1][ col-dot_s.second +1] =true;
    	// 			}
    	// 		}
    	// 		else
    	// 			continue;
    	// 	}
    	
    // 3.3:
    	for( int row = dot_s.first ; row <= dot_e.first -2 ; row++ )
    		for( int col = dot_s.second ; col <= dot_e.second -2 ; col++ )
    		{
    			if( !mask[row-dot_s.first][col-dot_s.second] && !mask[row-dot_s.first][col-dot_s.second+1] 
    				&& !mask[row-dot_s.first][col-dot_s.second+2] && mask[row-dot_s.first+1][col-dot_s.second] 
    				&& mask[row-dot_s.first+1][col-dot_s.second+1] && mask[row-dot_s.first+1][col-dot_s.second+2] 
    				&& mask[row-dot_s.first+2][col-dot_s.second] && !mask[row-dot_s.first+2][col-dot_s.second+1]
    				&& mask[row-dot_s.first+2][col-dot_s.second+2] )
    			{
    				breakpoint[row-dot_s.first +1][col-dot_s.second+1] = true;
    			}
    			else if( mask[row-dot_s.first][col-dot_s.second] && !mask[row-dot_s.first][col-dot_s.second+1] 
    				&& mask[row-dot_s.first][col-dot_s.second+2] && mask[row-dot_s.first+1][col-dot_s.second] 
    				&& mask[row-dot_s.first+1][col-dot_s.second+1] && mask[row-dot_s.first+1][col-dot_s.second+2] 
    				&& !mask[row-dot_s.first+2][col-dot_s.second] && !mask[row-dot_s.first+2][col-dot_s.second+1]
    				&& !mask[row-dot_s.first+2][col-dot_s.second+2] )
    			{
    				breakpoint[row-dot_s.first +1][col-dot_s.second+1] = true;
    			}
    			else 
    				continue;
    		}

    // 3.4:
    	for( int row = dot_s.first ; row <= dot_e.first -2 ; row++ )
    		for( int col = dot_s.second ; col <= dot_e.second -3 ; col++ )
    		{
    			if( mask[row-dot_s.first][col-dot_s.second] && !mask[row-dot_s.first][col-dot_s.second+1]
    				&& !mask[row-dot_s.first][col-dot_s.second+2] && mask[row-dot_s.first][col-dot_s.second+3]
    				&& mask[row-dot_s.first+1][col-dot_s.second] && mask[row-dot_s.first+1][col-dot_s.second+1]
    				&& mask[row-dot_s.first+1][col-dot_s.second+2] && mask[row-dot_s.first+1][col-dot_s.second+3]
    				&& !mask[row-dot_s.first+2][col-dot_s.second] && !mask[row-dot_s.first+2][col-dot_s.second+1]
    				&& !mask[row-dot_s.first+2][col-dot_s.second+2] && !mask[row-dot_s.first+2][col-dot_s.second+3] )
    			{
    				breakpoint[row-dot_s.first+1][col-dot_s.second+1] = true;
    				breakpoint[row-dot_s.first+1][col-dot_s.second+2] = true;
    			}
    			else if( !mask[row-dot_s.first][col-dot_s.second] && !mask[row-dot_s.first][col-dot_s.second+1]
    				&& !mask[row-dot_s.first][col-dot_s.second+2] && !mask[row-dot_s.first][col-dot_s.second+3]
    				&& mask[row-dot_s.first+1][col-dot_s.second] && mask[row-dot_s.first+1][col-dot_s.second+1]
    				&& mask[row-dot_s.first+1][col-dot_s.second+2] && mask[row-dot_s.first+1][col-dot_s.second+3]
    				&& mask[row-dot_s.first+2][col-dot_s.second] && !mask[row-dot_s.first+2][col-dot_s.second+1]
    				&& !mask[row-dot_s.first+2][col-dot_s.second+2] && mask[row-dot_s.first+2][col-dot_s.second+3] )
    			{
    				breakpoint[row-dot_s.first+1][col-dot_s.second+1] = true;
    				breakpoint[row-dot_s.first+1][col-dot_s.second+2] = true;
    			}
    			else
    				continue;

    		}




	// //-------//




 //    // Stage 4 : Filling holes 
 //    	for( int row = dot_s.first + 1 ; row <= dot_e.first -1 ; row++ )
 //    		for( int col = dot_s.second +1 ; col <= dot_e.second -1; col++ )
 //    		{
 //    			int num = 0;
 //    			for( int r = -1; r<=1 ; r++ )
 //    				for( int c = -1; c<=1 ; c++ )
 //    				{
 //    					if( r==0 && c==0)
 //    						continue;
 //    					num += ( mask[row+r-dot_s.first][col+c-dot_s.second] ? 1 : 0 );
 //    				}

 //    			if( num >= 7 )
 //    				mask[row-dot_s.first][col-dot_s.second] = true;
 //    		}


 //    //--------//
    



    // Stage 5 : Finish segmentation

    //---------//



    // Final process: binary mask		
		for( int r = 0 ; r < nRows ; r++ )
			for( int c = 0 ; c < nCols ; c++ )
			{
				if( mask[r][c] )
				{
					if( breakpoint[r][c] )
					{
						outImage.ptr<uchar>(dot_s.first + r)[ 3*(dot_s.second + c) ] = 0;
						outImage.ptr<uchar>(dot_s.first + r)[ 3*(dot_s.second + c) + 1] = 0;
						outImage.ptr<uchar>(dot_s.first + r)[ 3*(dot_s.second + c) + 2] = 255;
					}
					else
					{
						outImage.ptr<uchar>(dot_s.first + r)[ 3*(dot_s.second + c) ] = 0;
						outImage.ptr<uchar>(dot_s.first + r)[ 3*(dot_s.second + c) + 1] = 0;
						outImage.ptr<uchar>(dot_s.first + r)[ 3*(dot_s.second + c) + 2] = 0;
					}
				}
				else
				{
					outImage.ptr<uchar>(dot_s.first + r)[ 3*(dot_s.second + c) ] = 255;
					outImage.ptr<uchar>(dot_s.first + r)[ 3*(dot_s.second + c) + 1] = 255;
					outImage.ptr<uchar>(dot_s.first + r)[ 3*(dot_s.second + c) + 2] = 255;
				}
			}

		starts.push_back(dot_s);
		ends.push_back(dot_e);
	}

}


