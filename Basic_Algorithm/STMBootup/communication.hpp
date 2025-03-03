/**
 * @file communication.hpp
 * @brief Communication interface class definition
 * @author Claude
 * @date 2025-04-05
 */

 #ifndef COMMUNICATION_HPP
 #define COMMUNICATION_HPP
 
 #include <cstdint>
 
 /**
  * @class Communication
  * @brief Interface for communication with host
  */
 class Communication {
 public:
     /**
      * @brief Constructor
      */
     Communication();
     
     /**
      * @brief Destructor
      */
     ~Communication();
     
     /**
      * @brief Initialize communication interface
      * @return True if initialization successful, false otherwise
      */
     bool Initialize();
     
     /**
      * @brief Send data to host
      * @param data Data to send
      * @param size Size of data in bytes
      * @return True if send successful, false otherwise
      */
     bool SendData(const uint8_t* data, uint32_t size);
     
     /**
      * @brief Receive command from host
      * @param command Pointer to store command code
      * @param data Buffer to store command data
      * @param dataSize Pointer to store data size
      * @param timeout Timeout in milliseconds
      * @return True if receive successful, false otherwise
      */
     bool ReceiveCommand(uint8_t* Command, uint8_t* data, uint32_t* DataSize, 
                        uint32_t Timeout);
 
 private:
     // UART handle
     UART_HandleTypeDef Huart;
     
     // Receive buffer
     uint8_t RxBuffer[512];
     uint32_t RxIndex;
     
     // Calculate packet checksum
     uint8_t CalculateChecksum(const uint8_t* data, uint32_t size);
 };
 
 #endif // COMMUNICATION_HPP