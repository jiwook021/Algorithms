    /**
     * @file bootloader.cpp
     * @brief Bootloader class implementation
     * @author Claude
     * @date 2025-04-05
     */

    #include "bootloader.hpp"
    #include "stm32f4xx_hal.h"
    #include <cstring>
    
    // Define function pointer type for application entry point
    typedef void (*PFunction)(void);
    
    /**
     * @brief Bootloader constructor
     */
    Bootloader::Bootloader() : 
        State(BootloaderState::INIT),
        ReceivedBytes(0),
        ExpectedBytes(0),
        FirmwareBuffer(reinterpret_cast<uint8_t*>(FIRMWARE_BUFFER_ADDRESS)) {
        
        // Clear firmware header
        std::memset(&FirmwareHeader, 0, sizeof(FirmwareHeader));
    }
    
    /**
     * @brief Bootloader destructor
     */
    Bootloader::~Bootloader() {
        // Nothing to do here as memory is statically allocated
    }
    
    /**
     * @brief Initialize bootloader
     * @return True if initialization successful, false otherwise
     */
    bool Bootloader::Initialize() {
        // Initialize flash interface
        if (!FlashInterface.Initialize()) {
            return false;
        }
        
        // Initialize communication interface
        if (!Comm.Initialize()) {
            return false;
        }
        
        // Initialize CRC calculator
        CrcCalculator.Initialize();
        
        // Set state to initialization complete
        State = BootloaderState::CHECK_CONDITIONS;
        
        return true;
    }
    
    /**
     * @brief Run bootloader main process
     * @return Only returns on error
     */
    bool Bootloader::Run() {
        // Check if bootloader should run or jump to application
        if (State == BootloaderState::CHECK_CONDITIONS) {
            if (!CheckBootConditions()) {
                State = BootloaderState::JUMP_TO_APPLICATION;
            } else {
                State = BootloaderState::WAIT_FOR_COMMAND;
            }
        }
        
        // Main bootloader loop
        while (true) {
            switch (State) {
                case BootloaderState::WAIT_FOR_COMMAND: {
                    // Wait for command from host
                    uint8_t Command;
                    uint8_t data[256]; // Buffer for command data
                    uint32_t DataSize;
                    
                    if (Comm.ReceiveCommand(&Command, data, &DataSize, 5000)) { // 5 second timeout
                        if (!ProcessCommand(Command, data, DataSize)) {
                            // Handle command processing error
                            SendResponse(Command, ERR_INVALID_COMMAND);
                        }
                    } else {
                        // Check if we should timeout and jump to application
                        if (IsApplicationValid()) {
                            State = BootloaderState::JUMP_TO_APPLICATION;
                        }
                    }
                    break;
                }
                
                case BootloaderState::RECEIVING_FIRMWARE: {
                    // Firmware data processing is handled in ProcessCommand
                    break;
                }
                
                case BootloaderState::VERIFYING_FIRMWARE: {
                    if (VerifyFirmware()) {
                        State = BootloaderState::PROGRAMMING_FLASH;
                    } else {
                        SendResponse(CMD_FIRMWARE_VERIFY, ERR_VERIFICATION_FAILED);
                        State = BootloaderState::WAIT_FOR_COMMAND;
                    }
                    break;
                }
                
                case BootloaderState::PROGRAMMING_FLASH: {
                    if (ProgramFirmware()) {
                        // Send success response
                        SendResponse(CMD_FIRMWARE_VERIFY, ERR_NONE);
                        // Reset to apply new firmware
                        HAL_Delay(100); // Small delay to ensure response is sent
                        HAL_NVIC_SystemReset();
                    } else {
                        SendResponse(CMD_FIRMWARE_VERIFY, ERR_FLASH_ERROR);
                        State = BootloaderState::WAIT_FOR_COMMAND;
                    }
                    break;
                }
                
                case BootloaderState::JUMP_TO_APPLICATION: {
                    JumpToApplication();
                    // Should never return from JumpToApplication
                    // If we get here, there was an error
                    State = BootloaderState::ERROR;
                    break;
                }
                
                case BootloaderState::ERROR:
                default: {
                    // Handle error state
                    // In real bootloader, we might try to recover or reset
                    return false;
                }
            }
            
            // Feed watchdog and yield time to background tasks.
            // In production, replace HAL_Delay with a proper WDT kick:
            //   HAL_IWDG_Refresh(&hiwdg);
            HAL_Delay(10);
        }
        
        // Should never reach here
        return false;
    }
    
    /**
     * @brief Check if bootloader conditions are met
     * @return True if bootloader should run, false to jump to application
     * 
     * Time Complexity: O(1) - Constant time operations
     * Space Complexity: O(1) - Constant space usage
     */
    bool Bootloader::CheckBootConditions() {
        // Check if boot pin is pressed (active low)
        if (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_0) == GPIO_PIN_RESET) {
            return true; // Boot pin pressed, stay in bootloader
        }
        
        // Check if application is valid
        if (!IsApplicationValid()) {
            return true; // No valid application, stay in bootloader
        }
        
        // Check for firmware update flag in a designated flash location
        // This could be set by the application to request bootloader mode
        uint32_t UpdateFlag = *reinterpret_cast<volatile uint32_t*>(BOOTLOADER_START_ADDRESS + BOOTLOADER_SIZE - 4);
        if (UpdateFlag == 0xB00710AD) {  // "BOOTLOAD" marker
            // Clear the flag
            FlashInterface.EraseSector(BOOTLOADER_START_ADDRESS + BOOTLOADER_SIZE - 0x1000); // Last sector
            return true; // Update flag set, stay in bootloader
        }
        
        // No bootloader conditions met, jump to application
        return false;
    }
    
    /**
     * @brief Process received command
     * @param command Command code
     * @param data Command data
     * @param dataSize Size of command data
     * @return True if command processed successfully, false otherwise
     * 
     * Time Complexity: O(1) - Constant time for command processing
     * Space Complexity: O(1) - Constant space usage
     */
    bool Bootloader::ProcessCommand(uint8_t Command, const uint8_t* data, uint32_t DataSize) {
        switch (Command) {
            case CMD_PING: {
                // Simple ping command to check if bootloader is responsive
                return SendResponse(CMD_PING, ERR_NONE);
            }
            
            case CMD_GET_VERSION: {
                // Return bootloader version
                uint32_t Version = BOOTLOADER_VERSION;
                return SendResponse(CMD_GET_VERSION, ERR_NONE, 
                                reinterpret_cast<const uint8_t*>(&Version), sizeof(Version));
            }
            
            case CMD_START_FIRMWARE_UPDATE: {
                // Start firmware update process
                if (DataSize < sizeof(FirmwareHeader)) {
                    return false; // Invalid data size
                }
                
                // Copy firmware header
                std::memcpy(&FirmwareHeader, data, sizeof(FirmwareHeader));
                
                // Validate header
                if (FirmwareHeader.MagicNumber != FIRMWARE_MAGIC_NUMBER) {
                    return false; // Invalid magic number
                }
                
                // Check if firmware size is valid
                if (FirmwareHeader.FirmwareSize > FIRMWARE_BUFFER_SIZE || 
                    FirmwareHeader.FirmwareSize == 0) {
                    return false; // Invalid firmware size
                }
                
                // Check if destination address is valid
                if (FirmwareHeader.DestinationAddress < APPLICATION_START_ADDRESS || 
                    FirmwareHeader.DestinationAddress + FirmwareHeader.FirmwareSize > 
                    BOOTLOADER_START_ADDRESS + BOOTLOADER_SIZE + APPLICATION_SIZE) {
                    return false; // Invalid destination address
                }
                
                // Reset received bytes counter
                ReceivedBytes = 0;
                ExpectedBytes = FirmwareHeader.FirmwareSize;
                
                // Set state to receiving firmware
                State = BootloaderState::RECEIVING_FIRMWARE;
                
                // Send success response
                return SendResponse(CMD_START_FIRMWARE_UPDATE, ERR_NONE);
            }
            
            case CMD_FIRMWARE_DATA: {
                // Process firmware data packet
                if (State != BootloaderState::RECEIVING_FIRMWARE) {
                    return false; // Invalid state
                }
                
                return HandleFirmwareData(data, DataSize);
            }
            
            case CMD_FIRMWARE_VERIFY: {
                // Verify received firmware
                if (State != BootloaderState::RECEIVING_FIRMWARE || 
                    ReceivedBytes != ExpectedBytes) {
                    return false; // Invalid state or incomplete firmware
                }
                
                State = BootloaderState::VERIFYING_FIRMWARE;
                return true; // Verification will be handled in main loop
            }
            
            case CMD_RESET: {
                // Reset microcontroller
                SendResponse(CMD_RESET, ERR_NONE);
                HAL_Delay(100); // Small delay to ensure response is sent
                HAL_NVIC_SystemReset();
                return true; // Will never return from reset
            }
            
            default:
                return false; // Unknown command
        }
    }
    
    /**
     * @brief Handle firmware data packet
     * @param data Firmware data packet
     * @param dataSize Size of data packet
     * @return True if data processed successfully, false otherwise
     * 
     * Time Complexity: O(n) where n is dataSize - Linear time to copy data
     * Space Complexity: O(1) - Constant space usage (buffer is pre-allocated)
     */
    bool Bootloader::HandleFirmwareData(const uint8_t* data, uint32_t DataSize) {
        // Check if we can fit this data packet
        if (ReceivedBytes + DataSize > ExpectedBytes) {
            return false; // Too much data
        }
        
        // Copy data to firmware buffer
        std::memcpy(FirmwareBuffer + ReceivedBytes, data, DataSize);
        ReceivedBytes += DataSize;
        
        // Send response with current progress
        uint32_t Progress[2] = {ReceivedBytes, ExpectedBytes};
        return SendResponse(CMD_FIRMWARE_DATA, ERR_NONE, 
                        reinterpret_cast<const uint8_t*>(Progress), sizeof(Progress));
    }
    
    /**
     * @brief Verify received firmware
     * @return True if verification successful, false otherwise
     * 
     * Time Complexity: O(n) where n is firmwareSize - Linear time to calculate CRC
     * Space Complexity: O(1) - Constant space usage
     */
    bool Bootloader::VerifyFirmware() {
        // Verify firmware size
        if (ReceivedBytes != ExpectedBytes) {
            return false; // Size mismatch
        }
        
        // Calculate CRC
        uint32_t CalculatedCrc = CrcCalculator.Calculate(
            FirmwareBuffer, FirmwareHeader.FirmwareSize - sizeof(FirmwareHeader));
        
        // Verify CRC
        if (CalculatedCrc != FirmwareHeader.FirmwareCrc) {
            return false; // CRC mismatch
        }
        
        return true;
    }
    
    /**
     * @brief Program firmware to flash
     * @return True if programming successful, false otherwise
     * 
     * Time Complexity: O(n) where n is firmwareSize - Linear time to write data
     * Space Complexity: O(1) - Constant space usage
     */
    bool Bootloader::ProgramFirmware() {
        // Calculate number of sectors to erase
        uint32_t StartSector = FlashInterface.GetSectorNumber(FirmwareHeader.DestinationAddress);
        uint32_t EndAddress = FirmwareHeader.DestinationAddress + FirmwareHeader.FirmwareSize;
        uint32_t EndSector = FlashInterface.GetSectorNumber(EndAddress - 1);
        
        // Erase required flash sectors
        for (uint32_t Sector = StartSector; Sector <= EndSector; Sector++) {
            if (!FlashInterface.EraseSector(Sector)) {
                return false; // Erase failed
            }
        }
        
        // Program firmware to flash
        if (!FlashInterface.ProgramData(FirmwareHeader.DestinationAddress, 
                                    FirmwareBuffer, FirmwareHeader.FirmwareSize)) {
            return false; // Programming failed
        }
        
        // Verify programmed data against source buffer (volatile read from flash)
        for (uint32_t i = 0; i < FirmwareHeader.FirmwareSize; i++) {
            if (*reinterpret_cast<volatile uint8_t*>(FirmwareHeader.DestinationAddress + i) != FirmwareBuffer[i]) {
                return false; // Verification failed
            }
        }
        
        return true;
    }
    
    /**
     * @brief Jump to application
     * This function will not return if successful
     * 
     * Time Complexity: O(1) - Constant time operations
     * Space Complexity: O(1) - Constant space usage
     */
    void Bootloader::JumpToApplication() {
        // Disable all interrupts
        __disable_irq();
        
        // Clear all pending interrupts to prevent stale IRQs firing in the app
        for (uint32_t i = 0; i < 8; i++) {
            NVIC->ICER[i] = 0xFFFFFFFF;  // Disable all NVIC interrupts
            NVIC->ICPR[i] = 0xFFFFFFFF;  // Clear all pending interrupts
        }
        
        // Reset SysTick to prevent it from firing in the application
        SysTick->CTRL = 0;
        SysTick->LOAD = 0;
        SysTick->VAL  = 0;
        
        // Reset all peripherals
        HAL_DeInit();
        
        // Set the vector table offset to application start address
        SCB->VTOR = APPLICATION_START_ADDRESS;
        
        // Set the stack pointer to the application's stack pointer
        __set_MSP(*reinterpret_cast<volatile uint32_t*>(APPLICATION_START_ADDRESS));
        
        // Get application entry point
        PFunction JumpToApplication = reinterpret_cast<PFunction>(
            *reinterpret_cast<volatile uint32_t*>(APPLICATION_START_ADDRESS + 4));
        
        // Jump to application
        JumpToApplication();
        
        // Should never reach here
    }
    
    /**
     * @brief Send response to host
     * @param command Command being responded to
     * @param status Status code
     * @param data Additional data (optional)
     * @param dataSize Size of additional data (optional)
     * @return True if response sent successfully, false otherwise
     * 
     * Time Complexity: O(n) where n is dataSize - Linear time to send data
     * Space Complexity: O(n) - Temporary buffer for response
     */
    bool Bootloader::SendResponse(uint8_t Command, uint8_t Status, 
                                const uint8_t* data, uint32_t DataSize) {
        // Maximum response size (command + status + 256 bytes of data)
        uint8_t Response[258];
        
        // Build response
        Response[0] = Command;
        Response[1] = Status;
        
        // Add data if provided
        if (data != nullptr && DataSize > 0) {
            // Ensure we don't exceed buffer size
            uint32_t CopySize = (DataSize <= 256) ? DataSize : 256;
            std::memcpy(Response + 2, data, CopySize);
            return Comm.SendData(Response, 2 + CopySize);
        }
        
        // Send response
        return Comm.SendData(Response, 2);
    }
    
    /**
     * @brief Check if application is valid
     * @return True if application is valid, false otherwise
     * 
     * Time Complexity: O(1) - Constant time operations
     * Space Complexity: O(1) - Constant space usage
     */
    bool Bootloader::IsApplicationValid() {
        // Check if SP points to RAM (0x20000000 - 0x2001FFFF for STM32F4)
        uint32_t Sp = *reinterpret_cast<volatile uint32_t*>(APPLICATION_START_ADDRESS);
        if (Sp < 0x20000000 || Sp > 0x2001FFFF) {
            return false; // Invalid stack pointer
        }
        
        // Check if PC points to flash (0x08000000 - 0x080FFFFF for STM32F4)
        uint32_t Pc = *reinterpret_cast<volatile uint32_t*>(APPLICATION_START_ADDRESS + 4);
        if (Pc < (APPLICATION_START_ADDRESS + 1) || Pc > 0x080FFFFF) {
            return false; // Invalid program counter
        }
        
        // Check for application magic number if implemented
        // uint32_t magicNumber = *reinterpret_cast<uint32_t*>(APPLICATION_START_ADDRESS + 0x190);
        // if (magicNumber != APPLICATION_MAGIC_NUMBER) {
        //     return false; // Invalid magic number
        // }
        
        return true;
    }