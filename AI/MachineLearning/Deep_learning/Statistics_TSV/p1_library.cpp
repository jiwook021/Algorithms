/* p1_library.cpp
 *
 * Libraries needed for EECS 280 project 1
 * Project UID 5366c7e2b77742d5b2142097e51561a5
 *
 * by Andrew DeOrio <awdeorio@umich.edu>
 * 2015-04-28
 */

 ////////////////////////////// BEGIN csvstream.h //////////////////////////////
 //////////// GitHub hash 29c19d29854d6566d5452a10d12128737a61f327 /////////////
 /* -*- mode: c++ -*- */
#ifndef CSVSTREAM_H
#define CSVSTREAM_H
/* csvstream.h
 *
 * Andrew DeOrio <awdeorio@umich.edu>
 *
 * An easy-to-use CSV file parser for C++
 * https://github.com/awdeorio/csvstream
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <string>
#include <vector>
#include <map>
#include <regex>
#include <exception>

 // A custom exception type
class CsvstreamException : public std::exception {
public:
    const char* what() const noexcept override {
        return Msg.c_str();
    }
    const std::string Msg;
    CsvstreamException(const std::string& Msg) : Msg(Msg) {};
};


// csvstream interface
class Csvstream {
public:
    // Constructor from filename. Throws csvstream_exception if open fails.
    Csvstream(const std::string& Filename, char Delimiter = ',', bool Strict = true);

    // Constructor from stream
    Csvstream(std::istream& Is, char Delimiter = ',', bool Strict = true);

    // Destructor
    ~Csvstream();

    // Return false if an error flag on underlying stream is set
    explicit operator bool() const;

    // Return header processed by constructor
    std::vector<std::string> Getheader() const;

    // Stream extraction operator reads one row. Throws csvstream_exception if
    // the number of items in a row does not match the header.
    Csvstream& operator>> (std::map<std::string, std::string>& Row);

    // Stream extraction operator reads one row, keeping column order. Throws
    // csvstream_exception if the number of items in a row does not match the
    // header.
    Csvstream& operator>> (std::vector<std::pair<std::string, std::string> >& Row);

private:
    // Filename.  Used for error messages.
    std::string Filename;

    // File stream in CSV format, used when library is called with filename ctor
    std::ifstream Fin;

    // Stream in CSV format
    std::istream& Is;

    // Delimiter between columns
    char Delimiter;

    // Strictly enforce the number of values in each row.  Raise an exception if
    // a row contains too many values or too few compared to the header.  When
    // strict=false, ignore extra values and set missing values to empty string.
    bool Strict;

    // Line no in file.  Used for error messages
    size_t LineNo;

    // Store header column names
    std::vector<std::string> Header;

    // Process header, the first line of the file
    void ReadHeader();

    // Disable copying because copying streams is bad!
    Csvstream(const Csvstream&);
    Csvstream& operator= (const Csvstream&);
};


///////////////////////////////////////////////////////////////////////////////
// Implementation

// Read and tokenize one line from a stream
static bool ReadCsvLine(std::istream& Is,
    std::vector<std::string>& data,
    char Delimiter
) {

    // Add entry for first token, start with empty string
    data.clear();
    data.push_back(std::string());

    // Process one character at a time
    char c = '\0';
    enum State { BEGIN, QUOTED, QUOTED_ESCAPED, UNQUOTED, UNQUOTED_ESCAPED, END };
    State State = BEGIN;
    while (Is.get(c)) {
        switch (State) {
        case BEGIN:
            // We need this state transition to properly handle cases where nothing
            // is extracted.
            State = UNQUOTED;

            // Intended switch fallthrough.  Beginning with GCC7, this triggers an
            // error by default.  Disable the error for this specific line.
#if __GNUG__ && __GNUC__ >= 7
            [[fallthrough]];
#endif

        case UNQUOTED:
            if (c == '"') {
                // Change states when we see a double quote
                State = QUOTED;
            }
            else if (c == '\\') { //note this checks for a single backslash char
                State = UNQUOTED_ESCAPED;
                data.back() += c;
            }
            else if (c == Delimiter) {
                // If you see a delimiter, then start a new field with an empty string
                data.push_back("");
            }
            else if (c == '\n' || c == '\r') {
                // If you see a line ending *and it's not within a quoted token*, stop
                // parsing the line.  Works for UNIX (\n) and OSX (\r) line endings.
                // Consumes the line ending character.
                State = END;
            }
            else {
                // Append character to current token
                data.back() += c;
            }
            break;

        case UNQUOTED_ESCAPED:
            // If a character is escaped, add it no matter what.
            data.back() += c;
            State = UNQUOTED;
            break;

        case QUOTED:
            if (c == '"') {
                // Change states when we see a double quote
                State = UNQUOTED;
            }
            else if (c == '\\') {
                State = QUOTED_ESCAPED;
                data.back() += c;
            }
            else {
                // Append character to current token
                data.back() += c;
            }
            break;

        case QUOTED_ESCAPED:
            // If a character is escaped, add it no matter what.
            data.back() += c;
            State = QUOTED;
            break;

        case END:
            if (c == '\n') {
                // Handle second character of a Windows line ending (\r\n).  Do
                // nothing, only consume the character.
            }
            else {
                // If this wasn't a Windows line ending, then put character back for
                // the next call to read_csv_line()
                Is.unget();
            }

            // We're done with this line, so break out of both the switch and loop.
            goto MultilevelBreak; //This is a rare example where goto is OK
            break;

        default:
            assert(0);
            throw State;

        }//switch
    }//while

MultilevelBreak:
    // Clear the failbit if we extracted anything.  This is to mimic the behavior
    // of getline(), which will set the eofbit, but *not* the failbit if a partial
    // line is read.
    if (State != BEGIN) Is.clear();

    // Return status is the underlying stream's status
    return static_cast<bool>(Is);
}


Csvstream::Csvstream(const std::string& Filename, char Delimiter, bool Strict)
    : Filename(Filename),
    Is(Fin),
    Delimiter(Delimiter),
    Strict(Strict),
    LineNo(0) {

    // Open file
    Fin.open(Filename.c_str());
    if (!Fin.is_open()) {
        throw CsvstreamException("Error opening file: " + Filename);
    }

    // Process header
    ReadHeader();
}


Csvstream::Csvstream(std::istream& Is, char Delimiter, bool Strict)
    : Filename("[no filename]"),
    Is(Is),
    Delimiter(Delimiter),
    Strict(Strict),
    LineNo(0) {
    ReadHeader();
}


Csvstream::~Csvstream() {
    if (Fin.is_open()) Fin.close();
}


Csvstream::operator bool() const {
    return static_cast<bool>(Is);
}


std::vector<std::string> Csvstream::Getheader() const {
    return Header;
}


Csvstream& Csvstream::operator>> (std::map<std::string, std::string>& Row) {
    // Clear input row
    Row.clear();

    // Read one line from stream, bail out if we're at the end
    std::vector<std::string> data;
    if (!ReadCsvLine(Is, data, Delimiter)) return *this;
    LineNo += 1;

    // When strict mode is disabled, coerce the length of the data.  If data is
    // larger than header, discard extra values.  If data is smaller than header,
    // pad data with empty strings.
    if (!Strict) {
        data.resize(Header.size());
    }

    // Check length of data
    if (data.size() != Header.size()) {
        auto Msg = "Number of items in row does not match header. " +
            Filename + ":L" + std::to_string(LineNo) + " " +
            "header.size() = " + std::to_string(Header.size()) + " " +
            "row.size() = " + std::to_string(data.size()) + " "
            ;
        throw CsvstreamException(Msg);
    }

    // combine data and header into a row object
    for (size_t i = 0; i < data.size(); ++i) {
        Row[Header[i]] = data[i];
    }

    return *this;
}


Csvstream& Csvstream::operator>> (std::vector<std::pair<std::string, std::string> >& Row) {
    // Clear input row
    Row.clear();
    Row.resize(Header.size());

    // Read one line from stream, bail out if we're at the end
    std::vector<std::string> data;
    if (!ReadCsvLine(Is, data, Delimiter)) return *this;
    LineNo += 1;

    // When strict mode is disabled, coerce the length of the data.  If data is
    // larger than header, discard extra values.  If data is smaller than header,
    // pad data with empty strings.
    if (!Strict) {
        data.resize(Header.size());
    }

    // Check length of data
    if (Row.size() != Header.size()) {
        auto Msg = "Number of items in row does not match header. " +
            Filename + ":L" + std::to_string(LineNo) + " " +
            "header.size() = " + std::to_string(Header.size()) + " " +
            "row.size() = " + std::to_string(Row.size()) + " "
            ;
        throw CsvstreamException(Msg);
    }

    // combine data and header into a row object
    for (size_t i = 0; i < data.size(); ++i) {
        Row[i] = make_pair(Header[i], data[i]);
    }

    return *this;
}


void Csvstream::ReadHeader() {
    // read first line, which is the header
    if (!ReadCsvLine(Is, Header, Delimiter)) {
        throw CsvstreamException("error reading header");
    }
}

/////////////////////////////// END csvstream.h ///////////////////////////////


////////////////////////////// P1 library functions /////////////////////////////

#include "p1_library.h"
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
using namespace std;


void VSort(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
}


std::vector<double> ExtractColumn(std::string Filename,
    std::string ColumnName) {

    // open file
    ifstream Fin;
    Fin.open(Filename.c_str());
    if (!Fin.is_open()) {
        cout << "Error opening " << Filename << "\n";
        exit(1);
    }

    // use csvstream to read file
    Csvstream Csvin(Fin, '\t');

    // check for column name not found
    vector<string> Header = Csvin.Getheader();
    size_t Column = Header.size();
    for (size_t i = 0; i < Header.size(); ++i) {
        if (Header[i] == ColumnName) {
            Column = i;
            break;
        }
    }
    if (Column == Header.size()) {
        cout << "Error: column name " << ColumnName << " not found in "
            << Filename << "\n";
        Fin.close();
        exit(1);
    }

    // extract column of data
    vector<double> ColumnData;
    vector<pair<string, string>> Row;
    while (Csvin >> Row) {
        ColumnData.push_back(stod(Row[Column].second));
    }

    return ColumnData;
}

#endif