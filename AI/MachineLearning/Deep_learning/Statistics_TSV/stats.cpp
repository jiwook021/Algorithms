#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

int count(vector<double> v) {
    int CountNumber;
    CountNumber = v.size();
    return (CountNumber);
}

double Sum(vector<double> v) {
    double SumNumber = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        SumNumber = SumNumber + v.at(i);
    }
    return (SumNumber);
}

double Mean(vector<double> v) {
    int SumNumber = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        SumNumber = SumNumber + v.at(i);
    }
    int MeanNumber = SumNumber / v.size();
    return (MeanNumber);
}

double Median(vector<double> v) {
    std::sort(v.begin(), v.end());
    double MedianNumber;
    if (v.size() % 2 == 0) {
        MedianNumber = v.at((v.size() / 2) + 0.5) - v.at((v.size() / 2) - 0.5);
        return (MedianNumber);
    }
    else {
        MedianNumber = v.at((v.size() / 2) + 0.5);
        return (MedianNumber);
    }
}

double Mode(vector<double> v) {
    std::sort(v.begin(), v.end());
    int CountMax = 1;
    double ModeFinal = v.at(0);
    for (size_t i = 0; i < v.size(); ++i) {
        int count = 0;
        for (size_t j = 1; j < v.size(); ++j) {
            if (v.at(i) == v.at(j)) {
                ++count;
            }
        }
        if (count > CountMax) {
            CountMax = count;
            ModeFinal = v.at(i);
        }
    }
    return(ModeFinal);
}
double min(vector<double> v) {
    std::sort(v.begin(), v.end());
    double MinNumber = v.at(0);
    return (MinNumber);
}
double max(vector<double> v) {
    std::sort(v.begin(), v.end());
    double MaxNumber = v.at(v.size() - 1);
    return (MaxNumber);
}
double Stdev(vector<double> v) {
    double MeanN = Mean(v);
    double Total = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        double Numerator = (v.at(i) - MeanN) * (v.at(i) - MeanN);
        Total = Total + Numerator;
    }
    double StdevNumber = sqrt(Total / (v.size() - 1));
    return (StdevNumber);
}
double Percentile(vector<double> v, double p) {
    double PercentileNumber;
    double PercentileFinal;
    double Intpart;
    double Fractpart;
    std::sort(v.begin(), v.end());
    PercentileNumber = p * (v.size() - 1);
    Fractpart = modf(PercentileNumber, &Intpart);
    if (Intpart == v.size() - 1) {
        PercentileFinal = v.at(Intpart);
        return (PercentileFinal);
    }
    else {
        PercentileFinal = v.at(Intpart) + Fractpart * (v.at(Intpart + 1) - v.at(Intpart));
        return (PercentileFinal);
    }
}



vector<vector<double> > Summarize(vector<double> v) {
    std::sort(v.begin(), v.end());
    vector <vector <double> > Summary;
    int SummaryCount = 1;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v.at(i - 1) == v.at(i)) {
            SummaryCount++;
        }
        if (v.at(i - 1) != v.at(i)) {
            vector <double> Trial;
            Trial.push_back(v.at(i - 1));
            Trial.push_back(SummaryCount);
            Summary.push_back(Trial);
            SummaryCount = 1;
        }
        if (i == v.size() - 1) {
            vector <double> Trial;
            Trial.push_back(v.at(i));
            Trial.push_back(SummaryCount);
            Summary.push_back(Trial);
        }
    }
    return Summary;
}

