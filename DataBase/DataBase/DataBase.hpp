/**
 * @file DataBase.hpp
 * @brief Template-based file database for storing, searching, and modifying records.
 *
 * Provides a generic Database class that persists records to a binary file.
 * Record types must implement ReadFromFile, WriteToFile, ReadKey, Size,
 * and overloaded operators ==, >>, <<.
 */

#ifndef DATA_BASE_HPP
#define DATA_BASE_HPP

#include <fstream>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <optional>
#include <sstream>

/**
 * @class PersonalRecord
 * @brief Represents a personal information record with SSN, name, city, year, and salary.
 */
class PersonalRecord {
public:
    static constexpr int NAME_LEN = 10;
    static constexpr int CITY_LEN = 10;
    static constexpr int SSN_LEN = 9;

    /**
     * @brief Default constructor. Initializes empty record.
     */
    PersonalRecord();

    /**
     * @brief Parameterized constructor.
     * @param ssn Social Security Number (9 chars).
     * @param name Name (up to NAME_LEN chars).
     * @param city City (up to CITY_LEN chars).
     * @param year Birth year.
     * @param salary Salary.
     */
    PersonalRecord(const char* ssn, const char* name, const char* city,
                   int year, long salary);

    ~PersonalRecord();

    /// @brief Copy constructor.
    PersonalRecord(const PersonalRecord& other);

    /// @brief Copy assignment operator.
    PersonalRecord& operator=(const PersonalRecord& other);

    /**
     * @brief Write record to a binary file stream.
     * @param out Output file stream.
     */
    void WriteToFile(std::fstream& out) const;

    /**
     * @brief Read record from a binary file stream.
     * @param in Input file stream.
     * @return True if read succeeded.
     */
    bool ReadFromFile(std::fstream& in);

    /**
     * @brief Get the serialized size of this record in bytes.
     * @return Size in bytes.
     */
    int Size() const;

    /**
     * @brief Compare records by SSN.
     * @param other Record to compare against.
     * @return True if SSNs match.
     */
    bool operator==(const PersonalRecord& other) const;

    /**
     * @brief Get SSN as a string.
     * @return SSN string.
     */
    std::string GetSsn() const;

    /**
     * @brief Get name as a string.
     * @return Name string.
     */
    std::string GetName() const;

    /**
     * @brief Get city as a string.
     * @return City string.
     */
    std::string GetCity() const;

    /**
     * @brief Get birth year.
     * @return Year.
     */
    int GetYear() const;

    /**
     * @brief Get salary.
     * @return Salary.
     */
    long GetSalary() const;

    /**
     * @brief Format record as a human-readable string.
     * @return Formatted string.
     */
    std::string ToString() const;

    /**
     * @brief Output operator.
     */
    friend std::ostream& operator<<(std::ostream& out, const PersonalRecord& rec);

private:
    char ssn_[10];
    char* name_;
    char* city_;
    int year_;
    long salary_;
};

/**
 * @class DataBase
 * @brief Template file-based database that stores records in binary format.
 * @tparam T Record type that implements the required interface.
 *
 * Supports Add, Find, Modify, and GetAll operations on a binary file.
 */
template <class T>
class DataBase {
public:
    /**
     * @brief Construct a database with a given file name.
     * @param fileName Path to the database file.
     */
    explicit DataBase(const std::string& fileName);

    /**
     * @brief Add a record to the database file.
     * @param record Record to add.
     * @return True on success.
     */
    bool Add(const T& record);

    /**
     * @brief Find a record matching the given key record.
     * @param key Record whose key fields are used for search.
     * @return The matching record if found, std::nullopt otherwise.
     */
    std::optional<T> Find(const T& key);

    /**
     * @brief Modify a record in the database.
     * @param key Record to search for (by key).
     * @param updated The updated record to write.
     * @return True if the record was found and modified.
     */
    bool Modify(const T& key, const T& updated);

    /**
     * @brief Retrieve all records from the database.
     * @return Vector of all records.
     */
    std::vector<T> GetAll();

    /**
     * @brief Get the database file name.
     * @return File name string.
     */
    const std::string& GetFileName() const;

private:
    std::string fileName_;
};

// ============================================================================
// Template implementation (must be in header)
// ============================================================================

template <class T>
DataBase<T>::DataBase(const std::string& fileName)
    : fileName_(fileName) {}

template <class T>
bool DataBase<T>::Add(const T& record) {
    std::fstream file(fileName_, std::ios::in | std::ios::out |
                      std::ios::app | std::ios::binary);
    if (!file.is_open()) {
        // Try creating the file
        file.open(fileName_, std::ios::out | std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        file.close();
        file.open(fileName_, std::ios::in | std::ios::out |
                  std::ios::app | std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
    }
    file.seekp(0, std::ios::end);
    record.WriteToFile(file);
    file.close();
    return true;
}

template <class T>
std::optional<T> DataBase<T>::Find(const T& key) {
    std::fstream file(fileName_, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        return std::nullopt;
    }
    T tmp;
    while (tmp.ReadFromFile(file)) {
        if (tmp == key) {
            file.close();
            return tmp;
        }
    }
    file.close();
    return std::nullopt;
}

template <class T>
bool DataBase<T>::Modify(const T& key, const T& updated) {
    std::fstream file(fileName_, std::ios::in | std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    T tmp;
    while (tmp.ReadFromFile(file)) {
        if (tmp == key) {
            file.seekp(-updated.Size(), std::ios::cur);
            updated.WriteToFile(file);
            file.close();
            return true;
        }
    }
    file.close();
    return false;
}

template <class T>
std::vector<T> DataBase<T>::GetAll() {
    std::vector<T> records;
    std::fstream file(fileName_, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        return records;
    }
    T tmp;
    while (tmp.ReadFromFile(file)) {
        records.push_back(tmp);
    }
    file.close();
    return records;
}

template <class T>
const std::string& DataBase<T>::GetFileName() const {
    return fileName_;
}

#endif // DATA_BASE_HPP
