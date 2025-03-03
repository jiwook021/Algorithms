/**
 * @file communication.cpp
 * @brief Communication interface class implementation
 * @author Claude
 * @date 2025-04-05
 */

 #include "communication.hpp"
 #include "stm32f4xx_hal.h"
 
 // Communication protocol
 #define START_BYTE      0x7E
 #define END_BYTE        0x7F
 #define ESCAPE_BYTE     0x7D
 #define ESCAPE_MASK     0x20
 
 /**
  * @brief Communication constructor
  */
 Communication::Communication() : RxIndex(0) {
     // Clear receive buffer
     for (uint32_t i = 0; i < sizeof(RxBuffer); i++) {
         RxBuffer[i] = 0;
     }
 }
 
 /**
  * @brief Communication destructor
  */
 Communication::~Communication() {
     // Deinitialize UART
     HAL_UART_DeInit(&Huart);
 }
 
 /**
  * @brief Initialize communication interface
  * @return True if initialization successful, false otherwise
  * 
  * Time Complexity: O(1) - Constant time operations
  * Space Complexity: O(1) - Constant space usage
  */
 bool Communication::Initialize() {
     // Initialize UART
     Huart.Instance = USART2;
     Huart.Init.BaudRate = 115200;
     Huart.Init.WordLength = UART_WORDLENGTH_8B;
     Huart.Init.StopBits = UART_STOPBITS_1;
     Huart.Init.Parity = UART_PARITY_NONE;
     Huart.Init.Mode = UART_MODE_TX_RX;
     Huart.Init.HwFlowCtl = UART_HWCONTROL_NONE;
     Huart.Init.OverSampling = UART_OVERSAMPLING_16;
     
     // Enable USART2 clock
     __HAL_RCC_USART2_CLK_ENABLE();
     __HAL_RCC_GPIOA_CLK_ENABLE();
     
     // Configure GPIO pins for UART
     GPIO_InitTypeDef GPIO_InitStruct = {0};
     GPIO_InitStruct.Pin = GPIO_PIN_2 | GPIO_PIN_3;
     GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
     GPIO_InitStruct.Pull = GPIO_NOPULL;
     GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
     GPIO_InitStruct.Alternate = GPIO_AF7_USART2;
     HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
     
     return (HAL_UART_Init(&Huart) == HAL_OK);
 }
 
 /**
  * @brief Send data to host
  * @param data Data to send
  * @param size Size of data in bytes
  * @return True if send successful, false otherwise
  * 
  * Time Complexity: O(n) where n is size - Linear time to send data
  * Space Complexity: O(2n) - Worst case for escaped data
  */
 bool Communication::SendData(const uint8_t* data, uint32_t size) {
     // Fixed-size stack buffer — no heap allocation in embedded/RTOS context.
     // Max packet: START + escaped data (2× worst case) + escaped checksum (2) + END = size*2 + 4.
     static constexpr uint32_t MAX_PAYLOAD = 256;
     if (size > MAX_PAYLOAD) {
         return false; // Payload exceeds protocol maximum
     }
     uint8_t TxBuffer[MAX_PAYLOAD * 2 + 4];
     
     // Build packet
     uint32_t TxIndex = 0;
     
     // Start byte
     TxBuffer[TxIndex++] = START_BYTE;
     
     // Data with escape sequences
     for (uint32_t i = 0; i < size; i++) {
         // Check if byte needs escaping
         if (data[i] == START_BYTE || data[i] == END_BYTE || data[i] == ESCAPE_BYTE) {
             TxBuffer[TxIndex++] = ESCAPE_BYTE;
             TxBuffer[TxIndex++] = data[i] ^ ESCAPE_MASK;
         } else {
             TxBuffer[TxIndex++] = data[i];
         }
     }
     
     // Calculate checksum
     uint8_t Checksum = CalculateChecksum(data, size);
     
     // Add checksum with escape if needed
     if (Checksum == START_BYTE || Checksum == END_BYTE || Checksum == ESCAPE_BYTE) {
         TxBuffer[TxIndex++] = ESCAPE_BYTE;
         TxBuffer[TxIndex++] = Checksum ^ ESCAPE_MASK;
     } else {
         TxBuffer[TxIndex++] = Checksum;
     }
     
     // End byte
     TxBuffer[TxIndex++] = END_BYTE;
     
     // Send packet
     HAL_StatusTypeDef Status = HAL_UART_Transmit(&Huart, TxBuffer, TxIndex, 1000);
     
     return (Status == HAL_OK);
 }
 
 /**
  * @brief Receive command from host
  * @param command Pointer to store command code
  * @param data Buffer to store command data
  * @param dataSize Pointer to store data size
  * @param timeout Timeout in milliseconds
  * @return True if receive successful, false otherwise
  * 
  * Time Complexity: O(n) where n is packet size - Linear time to receive and process packet
  * Space Complexity: O(1) - Constant space usage (pre-allocated buffer)
  */
 bool Communication::ReceiveCommand(uint8_t* Command, uint8_t* data, uint32_t* DataSize, 
                                  uint32_t Timeout) {
     // Clear receive buffer
     RxIndex = 0;
     
     // Start time
     uint32_t StartTime = HAL_GetTick();
     
     // State machine variables
     bool InPacket = false;
     bool EscapeNext = false;
     uint32_t DataIndex = 0;
     uint8_t Checksum = 0;
     bool ChecksumReceived = false;
     
     // Receive loop
     while ((HAL_GetTick() - StartTime) < Timeout) {
         // Receive one byte with timeout
         uint8_t byte;
         HAL_StatusTypeDef Status = HAL_UART_Receive(&Huart, &byte, 1, 100);
         
         // Check for timeout
         if (Status == HAL_TIMEOUT) {
             continue; // No data received, continue waiting
         } else if (Status != HAL_OK) {
             return false; // Communication error
         }
         
         // Process received byte
         if (!InPacket) {
             if (byte == START_BYTE) {
                 InPacket = true;
                 DataIndex = 0;
                 ChecksumReceived = false;
             }
         } else {
             if (EscapeNext) {
                 // Process escaped byte
                 byte ^= ESCAPE_MASK;
                 EscapeNext = false;
                 
                 // Store byte
                 if (!ChecksumReceived) {
                     if (DataIndex == 0) {
                         *Command = byte; // First byte is command
                         DataIndex++;
                     } else {
                         data[DataIndex - 1] = byte; // Subsequent bytes are data
                         DataIndex++;
                     }
                 } else {
                     Checksum = byte;
                 }
             } else if (byte == ESCAPE_BYTE) {
                 EscapeNext = true;
             } else if (byte == END_BYTE) {
                 // End of packet
                 if (ChecksumReceived) {
                     // Verify checksum
                     uint8_t CalculatedChecksum;
                     if (DataIndex > 1) {
                         // Command + data — fixed buffer avoids non-standard VLA
                         uint8_t PacketData[512];
                         if (DataIndex > sizeof(PacketData)) {
                             InPacket = false;
                             continue;
                         }
                         PacketData[0] = *Command;
                         for (uint32_t i = 1; i < DataIndex; i++) {
                             PacketData[i] = data[i - 1];
                         }
                         CalculatedChecksum = CalculateChecksum(PacketData, DataIndex);
                     } else {
                         // Command only
                         CalculatedChecksum = CalculateChecksum(Command, 1);
                     }
                     
                     if (CalculatedChecksum == Checksum) {
                         // Checksum valid, packet received successfully
                         *DataSize = DataIndex - 1; // Subtract command byte
                         return true;
                     }
                 }
                 
                 // Reset state machine for next packet
                 InPacket = false;
             } else if (!ChecksumReceived) {
                 // Store regular byte
                 if (DataIndex == 0) {
                     *Command = byte; // First byte is command
                     DataIndex++;
                 } else {
                     data[DataIndex - 1] = byte; // Subsequent bytes are data
                     DataIndex++;
                 }
             } else {
                 Checksum = byte;
             }
             
             // Check if all data received
             if (DataIndex > 0 && !ChecksumReceived) {
                 ChecksumReceived = true;
             }
         }
     }
     
     // Timeout expired
     return false;
 }
 
 /**
  * @brief Calculate packet checksum
  * @param data Data to calculate checksum for
  * @param size Size of data in bytes
  * @return Calculated checksum
  * 
  * Time Complexity: O(n) where n is size - Linear time to calculate checksum
  * Space Complexity: O(1) - Constant space usage
  */
 uint8_t Communication::CalculateChecksum(const uint8_t* data, uint32_t size) {
     uint8_t Checksum = 0;
     
     for (uint32_t i = 0; i < size; i++) {
         Checksum ^= data[i]; // XOR all bytes
     }
     
     return Checksum;
 }