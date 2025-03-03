/**
 * @file main.cpp
 * @brief Entry point for the DataBase module.
 *
 * Demonstrates usage of the DataBase template class with PersonalRecord.
 * Provides an interactive menu for adding, finding, modifying, and
 * listing records.
 */

#include "DataBase.hpp"
#include <iostream>
#include <string>

/**
 * @brief Run the interactive database menu.
 * @param db Reference to the database instance.
 */
void RunMenu(DataBase<PersonalRecord>& db) {
    std::string option;

    std::cout << "=== Personal Record Database ===\n";
    std::cout << "File: " << db.GetFileName() << "\n\n";

    while (true) {
        std::cout << "1. Add record\n"
                  << "2. Find record by SSN\n"
                  << "3. Modify record\n"
                  << "4. List all records\n"
                  << "5. Exit\n"
                  << "Enter option: ";
        std::getline(std::cin, option);

        if (option == "1") {
            std::string ssn, name, city;
            int year;
            long salary;
            std::cout << "SSN (9 digits): ";
            std::getline(std::cin, ssn);
            std::cout << "Name: ";
            std::getline(std::cin, name);
            std::cout << "City: ";
            std::getline(std::cin, city);
            std::cout << "Birth year: ";
            std::cin >> year;
            std::cout << "Salary: ";
            std::cin >> salary;
            std::cin.ignore();

            PersonalRecord rec(ssn.c_str(), name.c_str(), city.c_str(), year, salary);
            if (db.Add(rec)) {
                std::cout << "Record added.\n";
            } else {
                std::cout << "Failed to add record.\n";
            }

        } else if (option == "2") {
            std::string ssn;
            std::cout << "Enter SSN to find: ";
            std::getline(std::cin, ssn);
            PersonalRecord key(ssn.c_str(), "", "", 0, 0);
            auto result = db.Find(key);
            if (result.has_value()) {
                std::cout << "Found: " << result.value() << "\n";
            } else {
                std::cout << "Record not found.\n";
            }

        } else if (option == "3") {
            std::string ssn, name, city;
            int year;
            long salary;
            std::cout << "SSN of record to modify: ";
            std::getline(std::cin, ssn);
            PersonalRecord key(ssn.c_str(), "", "", 0, 0);
            std::cout << "New Name: ";
            std::getline(std::cin, name);
            std::cout << "New City: ";
            std::getline(std::cin, city);
            std::cout << "New Year: ";
            std::cin >> year;
            std::cout << "New Salary: ";
            std::cin >> salary;
            std::cin.ignore();

            PersonalRecord updated(ssn.c_str(), name.c_str(), city.c_str(), year, salary);
            if (db.Modify(key, updated)) {
                std::cout << "Record modified.\n";
            } else {
                std::cout << "Record not found.\n";
            }

        } else if (option == "4") {
            auto records = db.GetAll();
            if (records.empty()) {
                std::cout << "No records.\n";
            } else {
                for (const auto& rec : records) {
                    std::cout << rec << "\n";
                }
            }

        } else if (option == "5") {
            break;

        } else {
            std::cout << "Invalid option.\n";
        }
        std::cout << "\n";
    }
}

int main() {
    DataBase<PersonalRecord> db("personal.dat");
    RunMenu(db);
    return 0;
}
