/**
 * @file test_DataBase.cpp
 * @brief Unit tests for the DataBase and PersonalRecord classes.
 *
 * Tests the public API with defined inputs and expected outputs.
 * Uses a temporary file that is cleaned up after each test.
 */

#include "DataBase.hpp"
#include <cassert>
#include <iostream>
#include <cstdio>
#include <filesystem>

/// @brief Temporary database file path used by tests.
static const std::string TEST_DB_FILE = "test_personal.dat";

/**
 * @brief Remove the test database file if it exists.
 */
void CleanUp() {
    std::remove(TEST_DB_FILE.c_str());
}

/**
 * @brief Test PersonalRecord construction and accessors.
 */
void TestPersonalRecordConstruction() {
    std::cout << "  TestPersonalRecordConstruction... ";

    PersonalRecord rec("123456789", "Alice", "Seattle", 1990, 75000);
    assert(rec.GetSsn() == "123456789");
    assert(rec.GetName() == "Alice");
    assert(rec.GetCity() == "Seattle");
    assert(rec.GetYear() == 1990);
    assert(rec.GetSalary() == 75000);

    std::cout << "PASSED\n";
}

/**
 * @brief Test PersonalRecord default constructor produces empty fields.
 */
void TestPersonalRecordDefault() {
    std::cout << "  TestPersonalRecordDefault... ";

    PersonalRecord rec;
    assert(rec.GetYear() == 0);
    assert(rec.GetSalary() == 0);

    std::cout << "PASSED\n";
}

/**
 * @brief Test PersonalRecord equality comparison by SSN.
 */
void TestPersonalRecordEquality() {
    std::cout << "  TestPersonalRecordEquality... ";

    PersonalRecord a("123456789", "Alice", "Seattle", 1990, 75000);
    PersonalRecord b("123456789", "Bob", "Denver", 1985, 60000);
    PersonalRecord c("987654321", "Alice", "Seattle", 1990, 75000);

    assert(a == b);   // Same SSN
    assert(!(a == c)); // Different SSN

    std::cout << "PASSED\n";
}

/**
 * @brief Test PersonalRecord copy constructor.
 */
void TestPersonalRecordCopy() {
    std::cout << "  TestPersonalRecordCopy... ";

    PersonalRecord original("111222333", "Carol", "Boston", 1988, 90000);
    PersonalRecord copy(original);

    assert(copy.GetSsn() == "111222333");
    assert(copy.GetName() == "Carol");
    assert(copy.GetCity() == "Boston");
    assert(copy.GetYear() == 1988);
    assert(copy.GetSalary() == 90000);
    assert(copy == original);

    std::cout << "PASSED\n";
}

/**
 * @brief Test adding and retrieving a single record.
 */
void TestAddAndFind() {
    std::cout << "  TestAddAndFind... ";
    CleanUp();

    DataBase<PersonalRecord> db(TEST_DB_FILE);
    PersonalRecord rec("111222333", "Bob", "Denver", 1985, 60000);

    assert(db.Add(rec));

    PersonalRecord key("111222333", "", "", 0, 0);
    auto result = db.Find(key);
    assert(result.has_value());
    assert(result->GetSsn() == "111222333");
    assert(result->GetName() == "Bob");
    assert(result->GetYear() == 1985);
    assert(result->GetSalary() == 60000);

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test finding a record that does not exist.
 */
void TestFindNotFound() {
    std::cout << "  TestFindNotFound... ";
    CleanUp();

    DataBase<PersonalRecord> db(TEST_DB_FILE);
    PersonalRecord rec("111222333", "Bob", "Denver", 1985, 60000);
    db.Add(rec);

    PersonalRecord key("999999999", "", "", 0, 0);
    auto result = db.Find(key);
    assert(!result.has_value());

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test adding multiple records and retrieving all.
 */
void TestGetAll() {
    std::cout << "  TestGetAll... ";
    CleanUp();

    DataBase<PersonalRecord> db(TEST_DB_FILE);

    db.Add(PersonalRecord("111111111", "Alice", "Seattle", 1990, 75000));
    db.Add(PersonalRecord("222222222", "Bob", "Denver", 1985, 60000));
    db.Add(PersonalRecord("333333333", "Carol", "Boston", 1992, 80000));

    auto records = db.GetAll();
    assert(records.size() == 3);
    assert(records[0].GetSsn() == "111111111");
    assert(records[1].GetSsn() == "222222222");
    assert(records[2].GetSsn() == "333333333");

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test modifying an existing record.
 */
void TestModify() {
    std::cout << "  TestModify... ";
    CleanUp();

    DataBase<PersonalRecord> db(TEST_DB_FILE);
    db.Add(PersonalRecord("444555666", "Dan", "Chicago", 1980, 50000));

    PersonalRecord key("444555666", "", "", 0, 0);
    PersonalRecord updated("444555666", "Daniel", "Austin", 1980, 95000);

    assert(db.Modify(key, updated));

    auto result = db.Find(key);
    assert(result.has_value());
    assert(result->GetName() == "Daniel");
    assert(result->GetCity() == "Austin");
    assert(result->GetSalary() == 95000);

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test modifying a non-existent record returns false.
 */
void TestModifyNotFound() {
    std::cout << "  TestModifyNotFound... ";
    CleanUp();

    DataBase<PersonalRecord> db(TEST_DB_FILE);
    db.Add(PersonalRecord("111111111", "Alice", "Seattle", 1990, 75000));

    PersonalRecord key("999999999", "", "", 0, 0);
    PersonalRecord updated("999999999", "Nobody", "Nowhere", 2000, 0);
    assert(!db.Modify(key, updated));

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test GetAll on empty database returns empty vector.
 */
void TestGetAllEmpty() {
    std::cout << "  TestGetAllEmpty... ";
    CleanUp();

    DataBase<PersonalRecord> db(TEST_DB_FILE);
    auto records = db.GetAll();
    assert(records.empty());

    CleanUp();
    std::cout << "PASSED\n";
}

/**
 * @brief Test ToString produces expected format.
 */
void TestToString() {
    std::cout << "  TestToString... ";

    PersonalRecord rec("123456789", "Alice", "Seattle", 1990, 75000);
    std::string str = rec.ToString();
    assert(str.find("123456789") != std::string::npos);
    assert(str.find("Alice") != std::string::npos);
    assert(str.find("Seattle") != std::string::npos);
    assert(str.find("1990") != std::string::npos);
    assert(str.find("75000") != std::string::npos);

    std::cout << "PASSED\n";
}

/**
 * @brief Main test runner.
 */
int main() {
    std::cout << "Running DataBase tests...\n";

    TestPersonalRecordConstruction();
    TestPersonalRecordDefault();
    TestPersonalRecordEquality();
    TestPersonalRecordCopy();
    TestAddAndFind();
    TestFindNotFound();
    TestGetAll();
    TestModify();
    TestModifyNotFound();
    TestGetAllEmpty();
    TestToString();

    std::cout << "\nAll tests PASSED.\n";
    return 0;
}
