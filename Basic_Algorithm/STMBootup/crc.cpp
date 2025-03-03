/**
 * @file crc.cpp
 * @brief CRC calculation class implementation
 * @author Claude
 * @date 2025-04-05
 */

 #include "crc.hpp"

 // CRC-32 polynomial (IEEE 802.3)
 #define CRC32_POLYNOMIAL 0xEDB88320
 
 /**
  * @brief CRC constructor
  */
 CRC::CRC() {
     // Initialize CRC table
     GenerateTable();
 }
 
 /**
  * @brief CRC destructor
  */
 CRC::~CRC() {
     // Nothing to clean up
 }
 
 /**
  * @brief Initialize CRC calculator
  * @return True if initialization successful, false otherwise
  * 
  * Time Complexity: O(1) - Constant time operations
  * Space Complexity: O(1) - Constant space usage
  */
 bool CRC::Initialize() {
     // Nothing to initialize
     return true;
 }
 
 /**
  * @brief Calculate CRC-32 for data
  * @param data Data to calculate CRC for
  * @param size Size of data in bytes
  * @return Calculated CRC-32 value
  * 
  * Time Complexity: O(n) where n is size - Linear time to calculate CRC
  * Space Complexity: O(1) - Constant space usage
  */
 uint32_t CRC::Calculate(const uint8_t* data, uint32_t size) {
     uint32_t Crc = 0xFFFFFFFF;
     
     for (uint32_t i = 0; i < size; i++) {
         uint8_t Index = (Crc ^ data[i]) & 0xFF;
         Crc = (Crc >> 8) ^ CrcTable[Index];
     }
     
     return Crc ^ 0xFFFFFFFF;
 }
 
 /**
  * @brief Generate CRC-32 lookup table
  * 
  * Time Complexity: O(1) - Constant time (256 iterations)
  * Space Complexity: O(1) - Constant space usage (256 entries)
  */
 void CRC::GenerateTable() {
     for (uint32_t i = 0; i < 256; i++) {
         uint32_t Crc = i;
         
         for (uint32_t j = 0; j < 8; j++) {
             if (Crc & 1) {
                 Crc = (Crc >> 1) ^ CRC32_POLYNOMIAL;
             } else {
                 Crc = Crc >> 1;
             }
         }
         
         CrcTable[i] = Crc;
     }
 }