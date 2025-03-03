// main.cpp 
// Project UID 5366c7e2b77742d5b2142097e51561a5

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <iomanip>
#include <limits>
#include <fstream>
#include <map>
#include <regex>
#include <exception>
#include <sstream>

#include "stats.h"
#include "p1_library.h"

using namespace std;



int main(int argc, char *argv[]) 
{
    string Filename = string(argv[1]);
    string Column = string(argv[2]);
    
    vector <double> ColumnData = ExtractColumn(Filename, Column);

    //summary
    vector <vector <double> > Summary = Summarize(ColumnData);
    cout << "Summary (value: frequency)" << endl;
    for (size_t i = 0; i < Summary.size(); ++i) {
        cout << Summary[i][0] << ": " << Summary[i][1] << endl;
    }
    cout << endl;
    //count
    cout << "count = " << count(ColumnData) << endl;
    //sum
    cout << "sum = " << Sum(ColumnData) << endl;
    //mean
    cout << "mean = " << Mean(ColumnData) << endl;
    //stdev
    cout << "stdev = " << Stdev(ColumnData) << endl;
    //median
    cout << "median = " << Median(ColumnData) << endl;
    //mode
    cout << "mode = " << Mode(ColumnData) << endl;
    //min
    cout << "min = " << min(ColumnData) << endl;
    //max
    cout << "max = " << max(ColumnData) << endl;
    //percentile
    cout << "  0th percentile = " << Percentile(ColumnData, 0) << endl;
    cout << " 25th percentile = " << Percentile(ColumnData, 0.25) << endl;
    cout << " 50th percentile = " << Percentile(ColumnData, 0.5) << endl;
    cout << " 75th percentile = " << Percentile(ColumnData, 0.75) << endl;
    cout << "100th percentile = " << Percentile(ColumnData, 1) << endl;

}