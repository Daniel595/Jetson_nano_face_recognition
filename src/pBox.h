#ifndef PBOX_H
#define PBOX_H
#include <stdlib.h>
#include <iostream>

using namespace std;
#define mydataFmt float


struct pBox
{
	mydataFmt *pdata;
	int width;
	int height;
	int channel;
};
struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    
    //[0][5] left eye x,y 
    //[1][6] right eye x,y
    //[2][7] nose x,y
    //[3][8] mouth left x,y
    //[4][9] mouth right x,y
    mydataFmt ppoint[10];
    mydataFmt regreCoord[4];
};

struct orderScore
{
    mydataFmt score;
    int oriOrder;
};
#endif