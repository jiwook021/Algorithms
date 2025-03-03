#ifndef STATS_H
#define STATS_H

#include <vector>


std::vector<std::vector<double> > Summarize(std::vector<double> v);

//EFFECTS: returns the count of the numbers in v
int count(std::vector<double> v);

//REQUIRES: v is not empty
//EFFECTS: returns the sum of the numbers in v
double Sum(std::vector<double> v);

//REQUIRES: v is not empty
//EFFECTS: returns the arithmetic mean of the numbers in v
double Mean(std::vector<double> v);

//REQUIRES: v is not empty
//EFFECTS: returns the sample median of the numbers in v
double Median(std::vector<double> v);

//REQUIRES: v is not empty
//EFFECTS: returns the mode of the numbers in v
double Mode(std::vector<double> v);

//REQUIRES: v is not empty
//EFFECTS: returns the min number in v
double min(std::vector<double> v);

//REQUIRES: v is not empty
//EFFECTS: returns the max number in v
double max(std::vector<double> v);

//REQUIRES: v contains at least 2 elements
//EFFECTS: returns the corrected sample standard deviation of the numbers in v
double Stdev(std::vector<double> v);

//REQUIRES: v is not empty
//          p is between 0 and 1, inclusive
//EFFECTS: returns the percentile p of the numbers in v like Microsoft Excel.
double Percentile(std::vector<double> v, double p);

#endif
#pragma once
