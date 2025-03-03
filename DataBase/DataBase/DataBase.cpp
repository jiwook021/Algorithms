/**
 * @file DataBase.cpp
 * @brief Implementation of the PersonalRecord class.
 *
 * The DataBase template class is fully implemented in the header.
 * This file provides the PersonalRecord implementation.
 */

#include "DataBase.hpp"
#include <cstring>
#include <sstream>

// ============================================================================
// PersonalRecord implementation
// ============================================================================

PersonalRecord::PersonalRecord()
    : year_(0), salary_(0) {
    std::memset(ssn_, 0, sizeof(ssn_));
    name_ = new char[NAME_LEN + 1]();
    city_ = new char[CITY_LEN + 1]();
}

PersonalRecord::PersonalRecord(const char* ssn, const char* name,
                               const char* city, int year, long salary)
    : year_(year), salary_(salary) {
    std::memset(ssn_, 0, sizeof(ssn_));
    std::strncpy(ssn_, ssn, SSN_LEN);
    ssn_[SSN_LEN] = '\0';

    name_ = new char[NAME_LEN + 1]();
    std::strncpy(name_, name, NAME_LEN);
    name_[NAME_LEN] = '\0';

    city_ = new char[CITY_LEN + 1]();
    std::strncpy(city_, city, CITY_LEN);
    city_[CITY_LEN] = '\0';
}

PersonalRecord::~PersonalRecord() {
    delete[] name_;
    delete[] city_;
}

PersonalRecord::PersonalRecord(const PersonalRecord& other)
    : year_(other.year_), salary_(other.salary_) {
    std::memcpy(ssn_, other.ssn_, sizeof(ssn_));
    name_ = new char[NAME_LEN + 1]();
    std::memcpy(name_, other.name_, NAME_LEN + 1);
    city_ = new char[CITY_LEN + 1]();
    std::memcpy(city_, other.city_, CITY_LEN + 1);
}

PersonalRecord& PersonalRecord::operator=(const PersonalRecord& other) {
    if (this != &other) {
        std::memcpy(ssn_, other.ssn_, sizeof(ssn_));
        std::memcpy(name_, other.name_, NAME_LEN + 1);
        std::memcpy(city_, other.city_, CITY_LEN + 1);
        year_ = other.year_;
        salary_ = other.salary_;
    }
    return *this;
}

void PersonalRecord::WriteToFile(std::fstream& out) const {
    out.write(ssn_, SSN_LEN);
    out.write(name_, NAME_LEN);
    out.write(city_, CITY_LEN);
    out.write(reinterpret_cast<const char*>(&year_), sizeof(int));
    out.write(reinterpret_cast<const char*>(&salary_), sizeof(long));
}

bool PersonalRecord::ReadFromFile(std::fstream& in) {
    in.read(ssn_, SSN_LEN);
    if (!in.good() || in.gcount() != SSN_LEN) {
        return false;
    }
    in.read(name_, NAME_LEN);
    in.read(city_, CITY_LEN);
    in.read(reinterpret_cast<char*>(&year_), sizeof(int));
    in.read(reinterpret_cast<char*>(&salary_), sizeof(long));
    ssn_[SSN_LEN] = '\0';
    name_[NAME_LEN] = '\0';
    city_[CITY_LEN] = '\0';
    return in.good();
}

int PersonalRecord::Size() const {
    return SSN_LEN + NAME_LEN + CITY_LEN + sizeof(int) + sizeof(long);
}

bool PersonalRecord::operator==(const PersonalRecord& other) const {
    return std::strncmp(ssn_, other.ssn_, SSN_LEN) == 0;
}

std::string PersonalRecord::GetSsn() const {
    return std::string(ssn_, SSN_LEN);
}

std::string PersonalRecord::GetName() const {
    return std::string(name_);
}

std::string PersonalRecord::GetCity() const {
    return std::string(city_);
}

int PersonalRecord::GetYear() const {
    return year_;
}

long PersonalRecord::GetSalary() const {
    return salary_;
}

std::string PersonalRecord::ToString() const {
    std::ostringstream oss;
    oss << "SSN=" << GetSsn()
        << " Name=" << GetName()
        << " City=" << GetCity()
        << " Year=" << year_
        << " Salary=" << salary_;
    return oss.str();
}

std::ostream& operator<<(std::ostream& out, const PersonalRecord& rec) {
    out << rec.ToString();
    return out;
}
