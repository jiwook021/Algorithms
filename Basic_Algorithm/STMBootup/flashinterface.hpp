/**
 * @file flash_interface.hpp
 * @brief Flash memory interface class definition
 * @author Claude
 * @date 2025-04-05
 */

 #ifndef FLASH_INTERFACE_HPP
 #define FLASH_INTERFACE_HPP
 
 #include <cstdint>
 
 /**
  * @class FlashInterface
  * @brief Interface for flash memory operations
  */
 class FlashInterface {
 public:
     /**
      * @brief Constructor
      */
     FlashInterface();
     
     /**
      * @brief Destructor
      */
     ~FlashInterface();
     
     /**
      * @brief Initialize flash interface
      * @return True if initialization successful, false otherwise
      */
     bool Initialize();
     
     /**
      * @brief Erase flash sector
      * @param sector Sector number to erase
      * @return True if erase successful, false otherwise
      */
     bool EraseSector(uint32_t Sector);
     
     /**
      * @brief Program data to flash
      * @param address Destination address in flash
      * @param data Source data
      * @param size Size of data in bytes
      * @return True if programming successful, false otherwise
      */
     bool ProgramData(uint32_t Address, const uint8_t* data, uint32_t size);
     
     /**
      * @brief Get sector number for address
      * @param address Flash address
      * @return Sector number
      */
     uint32_t GetSectorNumber(uint32_t Address);
     
     /**
      * @brief Get sector start address
      * @param sector Sector number
      * @return Sector start address
      */
     uint32_t GetSectorStartAddress(uint32_t Sector);
     
     /**
      * @brief Get sector size
      * @param sector Sector number
      * @return Sector size in bytes
      */
     uint32_t GetSectorSize(uint32_t Sector);
 
 private:
     // Unlock flash memory for write access
     bool UnlockFlash();
     
     // Lock flash memory to prevent accidental writes
     void LockFlash();
 };
 
 #endif // FLASH_INTERFACE_HPP