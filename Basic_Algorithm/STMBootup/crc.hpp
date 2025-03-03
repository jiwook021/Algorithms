/**
 * @file crc.hpp
 * @brief CRC calculation class definition
 * @author Claude
 * @date 2025-04-05
 */

 #ifndef CRC_HPP
 #define CRC_HPP
 
 #include <cstdint>
 
 /**
  * @class CRC
  * @brief Class for CRC-32 calculation
  */
 class CRC {
 public:
     /**
      * @brief Constructor
      */
     CRC();
     
     /**
      * @brief Destructor
      */
     ~CRC();
     
     /**
      * @brief Initialize CRC calculator
      * @return True if initialization successful, false otherwise
      */
     bool Initialize();
     
     /**
      * @brief Calculate CRC-32 for data
      * @param data Data to calculate CRC for
      * @param size Size of data in bytes
      * @return Calculated CRC-32 value
      */
     uint32_t Calculate(const uint8_t* data, uint32_t size);
 
 private:
     // CRC-32 lookup table
     uint32_t CrcTable[256];
     
     // Generate CRC-32 lookup table
     void GenerateTable();
 };
 
 #endif // CRC_HPP