/**
 * @file bootloader.hpp
 * @brief Bootloader class definition
 * @author Claude
 * @date 2025-04-05
 */

 #ifndef BOOTLOADER_HPP
 #define BOOTLOADER_HPP
 
 #include "flash_interface.hpp"
 #include "communication.hpp"
 #include "crc.hpp"
 #include <cstdint>
 
 // Memory addresses — constexpr for type safety (preferred over #define in C++)
 static constexpr uint32_t BOOTLOADER_VERSION        = 0x01000000;  // Version 1.0.0.0
 static constexpr uint32_t BOOTLOADER_START_ADDRESS  = 0x08000000;  // Start of flash memory
 static constexpr uint32_t BOOTLOADER_SIZE           = 0x10000;     // 64KB for bootloader
 static constexpr uint32_t APPLICATION_START_ADDRESS = BOOTLOADER_START_ADDRESS + BOOTLOADER_SIZE;
 static constexpr uint32_t APPLICATION_SIZE          = 0x70000;     // 448KB for application
 static constexpr uint32_t FIRMWARE_BUFFER_ADDRESS   = 0x20000000;  // Start of RAM
 static constexpr uint32_t FIRMWARE_BUFFER_SIZE      = 0x10000;     // 64KB buffer for firmware download
 
 // Magic number for firmware validation
 static constexpr uint32_t FIRMWARE_MAGIC_NUMBER     = 0xDEADBEEF;
 
 // Command codes
 static constexpr uint8_t CMD_PING                   = 0x01;
 static constexpr uint8_t CMD_GET_VERSION            = 0x02;
 static constexpr uint8_t CMD_START_FIRMWARE_UPDATE  = 0x03;
 static constexpr uint8_t CMD_FIRMWARE_DATA          = 0x04;
 static constexpr uint8_t CMD_FIRMWARE_VERIFY        = 0x05;
 static constexpr uint8_t CMD_RESET                  = 0x06;
 
 // Error codes
 static constexpr uint8_t ERR_NONE                   = 0x00;
 static constexpr uint8_t ERR_INVALID_COMMAND        = 0x01;
 static constexpr uint8_t ERR_FLASH_ERROR            = 0x02;
 static constexpr uint8_t ERR_VERIFICATION_FAILED    = 0x03;
 static constexpr uint8_t ERR_INVALID_ADDRESS        = 0x04;
 static constexpr uint8_t ERR_INVALID_SIZE           = 0x05;
 static constexpr uint8_t ERR_COMMUNICATION_ERROR    = 0x06;
 
 // Bootloader states
 enum class BootloaderState {
     INIT,
     CHECK_CONDITIONS,
     WAIT_FOR_COMMAND,
     RECEIVING_FIRMWARE,
     VERIFYING_FIRMWARE,
     PROGRAMMING_FLASH,
     JUMP_TO_APPLICATION,
     ERROR
 };
 
 // Firmware header structure
 struct FirmwareHeader {
     uint32_t MagicNumber;       // Magic number for validation
     uint32_t FirmwareSize;      // Size of firmware in bytes
     uint32_t FirmwareVersion;   // Version of firmware
     uint32_t FirmwareCrc;       // CRC of firmware
     uint32_t DestinationAddress; // Destination address in flash
     uint8_t Reserved[12];       // Reserved for future use
 };
 
 /**
  * @class Bootloader
  * @brief Main bootloader class responsible for update and boot process
  */
 class Bootloader {
 public:
     /**
      * @brief Constructor
      */
     Bootloader();
     
     /**
      * @brief Destructor
      */
     ~Bootloader();
     
     /**
      * @brief Initialize bootloader
      * @return True if initialization successful, false otherwise
      */
     bool Initialize();
     
     /**
      * @brief Run bootloader main process
      * This function will not return if successful
      * @return Only returns on error
      */
     bool Run();
 
 private:
     // Components
     FlashInterface FlashInterface;  // Flash memory interface
     Communication Comm;             // Communication interface
     CRC CrcCalculator;             // CRC calculator
     
     // State variables
     BootloaderState State;          // Current state of bootloader
     FirmwareHeader FirmwareHeader;  // Firmware header
     uint32_t ReceivedBytes;         // Number of bytes received
     uint32_t ExpectedBytes;         // Number of bytes expected
     uint8_t* FirmwareBuffer;        // Buffer for firmware download
     
     /**
      * @brief Check if bootloader conditions are met
      * @return True if bootloader should run, false to jump to application
      */
     bool CheckBootConditions();
     
     /**
      * @brief Process received command
      * @param command Command code
      * @param data Command data
      * @param dataSize Size of command data
      * @return True if command processed successfully, false otherwise
      */
     bool ProcessCommand(uint8_t Command, const uint8_t* data, uint32_t DataSize);
     
     /**
      * @brief Handle firmware data packet
      * @param data Firmware data packet
      * @param dataSize Size of data packet
      * @return True if data processed successfully, false otherwise
      */
     bool HandleFirmwareData(const uint8_t* data, uint32_t DataSize);
     
     /**
      * @brief Verify received firmware
      * @return True if verification successful, false otherwise
      */
     bool VerifyFirmware();
     
     /**
      * @brief Program firmware to flash
      * @return True if programming successful, false otherwise
      */
     bool ProgramFirmware();
     
     /**
      * @brief Jump to application
      * This function will not return if successful
      */
     void JumpToApplication();
     
     /**
      * @brief Send response to host
      * @param command Command being responded to
      * @param status Status code
      * @param data Additional data (optional)
      * @param dataSize Size of additional data (optional)
      * @return True if response sent successfully, false otherwise
      */
     bool SendResponse(uint8_t Command, uint8_t Status, 
                       const uint8_t* data = nullptr, uint32_t DataSize = 0);
     
     /**
      * @brief Check if application is valid
      * @return True if application is valid, false otherwise
      */
     bool IsApplicationValid();
 };
 
 #endif // BOOTLOADER_HPP