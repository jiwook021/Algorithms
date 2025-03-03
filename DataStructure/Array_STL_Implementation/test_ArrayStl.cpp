// array_stl_implementation_tests.cpp
// Compile: g++ -std=c++23 array_stl_implementation_tests.cpp -lgtest -lpthread -o array_stl_implementation_tests

#include "Array.hpp"

#include <array>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>

// ---- Capacity: size() returns the fixed number of elements ----

TEST(ArrayVsStdArray, Size_ReturnsFixedElementCount) {
    std::array<int, 5> StdArr{};
    Array<int, 5>  ImplArr{};

    EXPECT_EQ(ImplArr.Size(),     StdArr.size());
    EXPECT_EQ(ImplArr.max_size(), StdArr.max_size());
    EXPECT_EQ(ImplArr.Empty(),    StdArr.empty());
}

// ---- Capacity: empty() returns true for zero-size arrays ----

TEST(ArrayVsStdArray, Empty_TrueForZeroSizeArray) {
    std::array<int, 0> StdArr{};
    Array<int, 0>  ImplArr{};

    EXPECT_EQ(ImplArr.Size(),  StdArr.size());
    EXPECT_EQ(ImplArr.Empty(), StdArr.empty());
    EXPECT_TRUE(ImplArr.Empty());
    EXPECT_EQ(ImplArr.begin(), ImplArr.end());
}

// ---- Element Access: operator[] returns element at index without bounds checking ----

TEST(ArrayVsStdArray, SubscriptOperator_ReturnsElementAtIndex) {
    std::array<int, 4> StdArr  = {10, 20, 30, 40};
    Array<int, 4>  ImplArr = {10, 20, 30, 40};

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ImplArr[i], StdArr[i]);
    }
}

// ---- Element Access: at() returns element with bounds checking ----

TEST(ArrayVsStdArray, At_ReturnsElementWithBoundsCheck) {
    std::array<int, 3> StdArr  = {1, 2, 3};
    Array<int, 3>  ImplArr = {1, 2, 3};

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(ImplArr.At(i), StdArr.at(i));
    }
}

// ---- Element Access: at() throws std::out_of_range for invalid index ----

TEST(ArrayVsStdArray, At_ThrowsOutOfRangeForInvalidIndex) {
    std::array<int, 3> StdArr  = {1, 2, 3};
    Array<int, 3>  ImplArr = {1, 2, 3};

    EXPECT_THROW(StdArr.at(3),  std::out_of_range);
    EXPECT_THROW(ImplArr.At(3), std::out_of_range);
    EXPECT_THROW(ImplArr.At(100), std::out_of_range);
}

// ---- Element Access: front() returns first element, back() returns last ----

TEST(ArrayVsStdArray, FrontBack_ReturnFirstAndLastElement) {
    std::array<int, 5> StdArr  = {5, 10, 15, 20, 25};
    Array<int, 5>  ImplArr = {5, 10, 15, 20, 25};

    EXPECT_EQ(ImplArr.Front(), StdArr.front());
    EXPECT_EQ(ImplArr.Back(),  StdArr.back());
    EXPECT_EQ(ImplArr.Front(), 5);
    EXPECT_EQ(ImplArr.Back(), 25);
}

// ---- Element Access: front()/back() are mutable references ----

TEST(ArrayVsStdArray, FrontBack_AreMutableReferences) {
    std::array<int, 3> StdArr  = {1, 2, 3};
    Array<int, 3>  ImplArr = {1, 2, 3};

    StdArr.front() = 99;
    ImplArr.Front() = 99;
    StdArr.back() = 77;
    ImplArr.Back() = 77;

    EXPECT_EQ(ImplArr.Front(), StdArr.front());
    EXPECT_EQ(ImplArr.Back(),  StdArr.back());
}

// ---- Element Access: data() returns pointer to underlying contiguous storage ----

TEST(ArrayVsStdArray, Data_ReturnsPointerToContiguousStorage) {
    std::array<int, 3> StdArr  = {7, 8, 9};
    Array<int, 3>  ImplArr = {7, 8, 9};

    EXPECT_EQ(*ImplArr.data(), *StdArr.data());
    EXPECT_EQ(ImplArr.data()[0], ImplArr.Front());
    EXPECT_EQ(ImplArr.data()[2], ImplArr.Back());
    EXPECT_EQ(ImplArr.data() + ImplArr.Size(), ImplArr.end());
}

// ---- Iterators: begin()/end() define a valid forward range ----

TEST(ArrayVsStdArray, BeginEnd_DefineValidForwardRange) {
    std::array<int, 4> StdArr  = {1, 2, 3, 4};
    Array<int, 4>  ImplArr = {1, 2, 3, 4};

    EXPECT_TRUE(std::equal(ImplArr.begin(), ImplArr.end(),
                           StdArr.begin(),  StdArr.end()));
    EXPECT_EQ(ImplArr.end() - ImplArr.begin(), 4);
}

// ---- Iterators: cbegin()/cend() provide const access ----

TEST(ArrayVsStdArray, CbeginCend_ProvideConstAccess) {
    const Array<int, 3> ImplArr = {10, 20, 30};
    const std::array<int, 3> StdArr = {10, 20, 30};

    EXPECT_TRUE(std::equal(ImplArr.cbegin(), ImplArr.cend(),
                           StdArr.cbegin(),  StdArr.cend()));
}

// ---- Iterators: rbegin()/rend() iterate in reverse order ----

TEST(ArrayVsStdArray, RbeginRend_IterateInReverse) {
    std::array<int, 5> StdArr  = {1, 2, 3, 4, 5};
    Array<int, 5>  ImplArr = {1, 2, 3, 4, 5};

    std::vector<int> StdRev(StdArr.rbegin(), StdArr.rend());
    std::vector<int> ImplRev(ImplArr.rbegin(), ImplArr.rend());
    EXPECT_EQ(ImplRev, StdRev);
    EXPECT_EQ(ImplRev, (std::vector<int>{5, 4, 3, 2, 1}));
}

// ---- Iterators: crbegin()/crend() provide const reverse access ----

TEST(ArrayVsStdArray, CrbeginCrend_ProvideConstReverseAccess) {
    const std::array<int, 4> StdArr  = {9, 8, 7, 6};
    const Array<int, 4>  ImplArr = {9, 8, 7, 6};

    std::vector<int> StdRev(StdArr.crbegin(), StdArr.crend());
    std::vector<int> ImplRev(ImplArr.crbegin(), ImplArr.crend());
    EXPECT_EQ(ImplRev, StdRev);
}

// ---- Operations: fill() sets all elements to the given value ----

TEST(ArrayVsStdArray, Fill_SetsAllElementsToValue) {
    std::array<int, 5> StdArr{};
    Array<int, 5>  ImplArr{};

    StdArr.fill(42);
    ImplArr.fill(42);

    EXPECT_TRUE(std::equal(ImplArr.begin(), ImplArr.end(),
                           StdArr.begin(),  StdArr.end()));
    for (auto v : ImplArr) EXPECT_EQ(v, 42);
}

// ---- Operations: swap() exchanges contents with another array of same type/size ----

TEST(ArrayVsStdArray, Swap_ExchangesContentsBetweenArrays) {
    std::array<int, 3> StdA = {1, 2, 3}, StdB = {4, 5, 6};
    Array<int, 3>  ImplA = {1, 2, 3}, ImplB = {4, 5, 6};

    StdA.swap(StdB);
    ImplA.Swap(ImplB);

    EXPECT_TRUE(std::equal(ImplA.begin(), ImplA.end(), StdA.begin(), StdA.end()));
    EXPECT_TRUE(std::equal(ImplB.begin(), ImplB.end(), StdB.begin(), StdB.end()));
}

// ---- Operations: free swap() enables ADL-based swapping ----

TEST(ArrayVsStdArray, FreeSwap_EnablesADLBasedSwapping) {
    std::array<int, 3> StdA = {10, 20, 30}, StdB = {40, 50, 60};
    Array<int, 3>  ImplA = {10, 20, 30}, ImplB = {40, 50, 60};

    std::swap(StdA, StdB);
    std::swap(ImplA, ImplB);

    EXPECT_TRUE(std::equal(ImplA.begin(), ImplA.end(), StdA.begin(), StdA.end()));
    EXPECT_TRUE(std::equal(ImplB.begin(), ImplB.end(), StdB.begin(), StdB.end()));
}

// ---- STL Compatibility: works with std::sort ----

TEST(ArrayVsStdArray, StdSort_ProducesSameOrderAsStdArray) {
    std::array<int, 6> StdArr  = {5, 3, 1, 4, 2, 6};
    Array<int, 6>  ImplArr = {5, 3, 1, 4, 2, 6};

    std::sort(StdArr.begin(), StdArr.end());
    std::sort(ImplArr.begin(), ImplArr.end());

    EXPECT_TRUE(std::equal(ImplArr.begin(), ImplArr.end(),
                           StdArr.begin(),  StdArr.end()));
}

// ---- STL Compatibility: works with std::accumulate ----

TEST(ArrayVsStdArray, StdAccumulate_ProducesSameSum) {
    std::array<int, 4> StdArr  = {1, 2, 3, 4};
    Array<int, 4>  ImplArr = {1, 2, 3, 4};

    int StdSum  = std::accumulate(StdArr.begin(),  StdArr.end(),  0);
    int ImplSum = std::accumulate(ImplArr.begin(), ImplArr.end(), 0);

    EXPECT_EQ(ImplSum, StdSum);
    EXPECT_EQ(ImplSum, 10);
}

// ---- STL Compatibility: works with range-based for loop ----

TEST(ArrayVsStdArray, RangeFor_IteratesSameAsStdArray) {
    std::array<int, 3> StdArr  = {100, 200, 300};
    Array<int, 3>  ImplArr = {100, 200, 300};

    std::vector<int> StdVals, ImplVals;
    for (auto v : StdArr)  StdVals.push_back(v);
    for (auto v : ImplArr) ImplVals.push_back(v);

    EXPECT_EQ(ImplVals, StdVals);
}

// ---- STL Compatibility: mutable iteration via range-based for ----

TEST(ArrayVsStdArray, RangeFor_MutableIterationModifiesElements) {
    std::array<int, 3> StdArr  = {1, 2, 3};
    Array<int, 3>  ImplArr = {1, 2, 3};

    for (auto& v : StdArr)  v *= 10;
    for (auto& v : ImplArr) v *= 10;

    EXPECT_TRUE(std::equal(ImplArr.begin(), ImplArr.end(),
                           StdArr.begin(),  StdArr.end()));
}

// ---- STL Compatibility: works with std::find ----

TEST(ArrayVsStdArray, StdFind_LocatesElementSameAsStdArray) {
    std::array<int, 5> StdArr  = {10, 20, 30, 40, 50};
    Array<int, 5>  ImplArr = {10, 20, 30, 40, 50};

    auto StdIt  = std::find(StdArr.begin(), StdArr.end(), 30);
    auto ImplIt = std::find(ImplArr.begin(), ImplArr.end(), 30);

    EXPECT_EQ(StdIt - StdArr.begin(), ImplIt - ImplArr.begin());

    auto StdMiss  = std::find(StdArr.begin(), StdArr.end(), 99);
    auto ImplMiss = std::find(ImplArr.begin(), ImplArr.end(), 99);
    EXPECT_EQ(StdMiss, StdArr.end());
    EXPECT_EQ(ImplMiss, ImplArr.end());
}

// ---- STL Compatibility: works with std::reverse ----

TEST(ArrayVsStdArray, StdReverse_ReversesElementsSameAsStdArray) {
    std::array<int, 5> StdArr  = {1, 2, 3, 4, 5};
    Array<int, 5>  ImplArr = {1, 2, 3, 4, 5};

    std::reverse(StdArr.begin(), StdArr.end());
    std::reverse(ImplArr.begin(), ImplArr.end());

    EXPECT_TRUE(std::equal(ImplArr.begin(), ImplArr.end(),
                           StdArr.begin(),  StdArr.end()));
}

// ---- Aggregate Init: supports brace initialization like std::array ----

TEST(ArrayVsStdArray, AggregateInit_SupportsBraceInitialization) {
    std::array<int, 4> StdArr  = {1, 2, 3, 4};
    Array<int, 4>  ImplArr = {1, 2, 3, 4};

    EXPECT_TRUE(std::equal(ImplArr.begin(), ImplArr.end(),
                           StdArr.begin(),  StdArr.end()));
}

// ---- Different Types: works correctly with double ----

TEST(ArrayVsStdArray, DoubleType_BehavesSameAsStdArray) {
    std::array<double, 3> StdArr  = {1.1, 2.2, 3.3};
    Array<double, 3>  ImplArr = {1.1, 2.2, 3.3};

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(ImplArr[i], StdArr[i]);
    }
}

// ---- Different Types: works correctly with char ----

TEST(ArrayVsStdArray, CharType_BehavesSameAsStdArray) {
    std::array<char, 5> StdArr  = {'h', 'e', 'l', 'l', 'o'};
    Array<char, 5>  ImplArr = {'h', 'e', 'l', 'l', 'o'};

    EXPECT_TRUE(std::equal(ImplArr.begin(), ImplArr.end(),
                           StdArr.begin(),  StdArr.end()));
    EXPECT_EQ(ImplArr.Front(), StdArr.front());
    EXPECT_EQ(ImplArr.Back(),  StdArr.back());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
