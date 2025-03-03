/**
 * @file flash_interface.cpp
 * @brief Flash memory interface class implementation
 * @author Claude
 * @date 2025-04-05
 */

 #include "flash_interface.hpp"
 #include "stm32f4xx_hal.h"
 
 /**
  * @brief FlashInterface constructor
  */
 FlashInterface::FlashInterface() {
     // Nothing to initialize here
 }
 
 /**
  * @brief FlashInterface destructor
  */
 FlashInterface::~FlashInterface() {
     // Ensure flash is locked on destruction
     LockFlash();
 }
 
 /**
  * @brief Initialize flash interface
  * @return True if initialization successful, false otherwise
  * 
  * Time Complexity: O(1) - Constant time operations
  * Space Complexity: O(1) - Constant space usage
  */
 bool FlashInterface::Initialize() {
     // Nothing to initialize for flash interface
     return true;
 }
 
 /**
  * @brief Erase flash sector
  * @param sector Sector number to erase
  * @return True if erase successful, false otherwise
  * 
  * Time Complexity: O(1) - Constant time for erase operation (hardware dependent)
  * Space Complexity: O(1) - Constant space usage
  */
 bool FlashInterface::EraseSector(uint32_t Sector) {
     // Unlock flash
     if (!UnlockFlash()) {
         return false;
     }
     
     // Initialize erase structure
     FLASH_EraseInitTypeDef EraseInit;
     EraseInit.TypeErase = FLASH_TYPEERASE_SECTORS;
     EraseInit.VoltageRange = FLASH_VOLTAGE_RANGE_3; // 2.7V to 3.6V
     EraseInit.Sector = Sector;
     EraseInit.NbSectors = 1;
     
     // Perform erase operation
     uint32_t SectorError = 0;
     HAL_StatusTypeDef Status = HAL_FLASHEx_Erase(&EraseInit, &SectorError);
     
     // Lock flash
     LockFlash();
     
     return (Status == HAL_OK && SectorError == 0xFFFFFFFF);
 }
 
 /**
  * @brief Program data to flash
  * @param address Destination address in flash
  * @param data Source data
  * @param size Size of data in bytes
  * @return True if programming successful, false otherwise
  * 
  * Time Complexity: O(n) where n is size - Linear time to write data
  * Space Complexity: O(1) - Constant space usage
  */
 bool FlashInterface::ProgramData(uint32_t Address, const uint8_t* data, uint32_t size) {
     // Check alignment
     if (Address % 4 != 0) {
         return false; // Address must be word-aligned
     }
     
     // Unlock flash
     if (!UnlockFlash()) {
         return false;
     }
     
     // Program data in 32-bit words
     bool Success = true;
     for (uint32_t i = 0; i < size; i += 4) {
         uint32_t Word = 0;
         
         // Build word from bytes (little-endian)
         for (uint32_t j = 0; j < 4 && (i + j) < size; j++) {
             Word |= static_cast<uint32_t>(data[i + j]) << (j * 8);
         }
         
         // Program word
         HAL_StatusTypeDef Status = HAL_FLASH_Program(FLASH_TYPEPROGRAM_WORD, 
                                                     Address + i, Word);
         if (Status != HAL_OK) {
             Success = false;
             break;
         }
     }
     
     // Lock flash
     LockFlash();
     
     return Success;
 }
 
 /**
  * @brief Get sector number for address
  * @param address Flash address
  * @return Sector number
  * 
  * Time Complexity: O(1) - Constant time operations
  * Space Complexity: O(1) - Constant space usage
  */
 uint32_t FlashInterface::GetSectorNumber(uint32_t Address) {
     // STM32F4 flash memory layout (non-uniform sectors)
     if (Address < 0x08003FFF) {
         return FLASH_SECTOR_0; // 16KB
     } else if (Address < 0x08007FFF) {
         return FLASH_SECTOR_1; // 16KB
     } else if (Address < 0x0800BFFF) {
         return FLASH_SECTOR_2; // 16KB
     } else if (Address < 0x0800FFFF) {
         return FLASH_SECTOR_3; // 16KB
     } else if (Address < 0x0801FFFF) {
         return FLASH_SECTOR_4; // 64KB
     } else if (Address < 0x0803FFFF) {
         return FLASH_SECTOR_5; // 128KB
     } else if (Address < 0x0805FFFF) {
         return FLASH_SECTOR_6; // 128KB
     } else if (Address < 0x0807FFFF) {
         return FLASH_SECTOR_7; // 128KB
     } else if (Address < 0x0809FFFF) {
         return FLASH_SECTOR_8; // 128KB
     } else if (Address < 0x080BFFFF) {
         return FLASH_SECTOR_9; // 128KB
     } else if (Address < 0x080DFFFF) {
         return FLASH_SECTOR_10; // 128KB
     } else {
         return FLASH_SECTOR_11; // 128KB
     }
 }
 
 /**
  * @brief Get sector start address
  * @param sector Sector number
  * @return Sector start address
  * 
  * Time Complexity: O(1) - Constant time operations
  * Space Complexity: O(1) - Constant space usage
  */
 uint32_t FlashInterface::GetSectorStartAddress(uint32_t Sector) {
     switch (Sector) {
         case FLASH_SECTOR_0: return 0x08000000;
         case FLASH_SECTOR_1: return 0x08004000;
         case FLASH_SECTOR_2: return 0x08008000;
         case FLASH_SECTOR_3: return 0x0800C000;
         case FLASH_SECTOR_4: return 0x08010000;
         case FLASH_SECTOR_5: return 0x08020000;
         case FLASH_SECTOR_6: return 0x08040000;
         case FLASH_SECTOR_7: return 0x08060000;
         case FLASH_SECTOR_8: return 0x08080000;
         case FLASH_SECTOR_9: return 0x080A0000;
         case FLASH_SECTOR_10: return 0x080C0000;
         case FLASH_SECTOR_11: return 0x080E0000;
         default: return 0xFFFFFFFF; // Invalid sector
     }
 }
 
 /**
  * @brief Get sector size
  * @param sector Sector number
  * @return Sector size in bytes
  * 
  * Time Complexity: O(1) - Constant time operations
  * Space Complexity: O(1) - Constant space usage
  */
 uint32_t FlashInterface::GetSectorSize(uint32_t Sector) {
     switch (Sector) {
         case FLASH_SECTOR_0:
         case FLASH_SECTOR_1:
         case FLASH_SECTOR_2:
         case FLASH_SECTOR_3:
             return 0x4000; // 16KB
         case FLASH_SECTOR_4:
             return 0x10000; // 64KB
         case FLASH_SECTOR_5:
         case FLASH_SECTOR_6:
         case FLASH_SECTOR_7:
         case FLASH_SECTOR_8:
         case FLASH_SECTOR_9:
         case FLASH_SECTOR_10:
         case FLASH_SECTOR_11:
             return 0x20000; // 128KB
         default:
             return 0; // Invalid sector
     }
 }
 
 /**
  * @brief Unlock flash memory for write access
  * @return True if unlock successful, false otherwise
  * 
  * Time Complexity: O(1) - Constant time operations
  * Space Complexity: O(1) - Constant space usage
  */
 bool FlashInterface::UnlockFlash() {
     HAL_StatusTypeDef Status = HAL_FLASH_Unlock();
     
     // Clear all flash flags
     __HAL_FLASH_CLEAR_FLAG(FLASH_FLAG_EOP | FLASH_FLAG_OPERR | FLASH_FLAG_WRPERR | 
                          FLASH_FLAG_PGAERR | FLASH_FLAG_PGPERR | FLASH_FLAG_PGSERR);
     
     return (Status == HAL_OK);
 }
 
 /**
  * @brief Lock flash memory to prevent accidental writes
  * 
  * Time Complexity: O(1) - Constant time operations
  * Space Complexity: O(1) - Constant space usage
  */
 void FlashInterface::LockFlash() {
     HAL_FLASH_Lock();
 }