/**
 * @file main.cpp
 * @brief Main entry point for the STM32 Cortex-M Bootloader
 * @author Claude
 * @date 2025-04-05
 */

 #include "bootloader.hpp"
 #include "stm32f4xx_hal.h"
 #include <cstdint>
 
 // Global bootloader instance
 static Bootloader Bootloader;
 
 /**
  * @brief System Clock Configuration
  * Configures the system clock to the maximum frequency for the device
  * @return None
  */
 void SystemClock_Config(void) {
     RCC_OscInitTypeDef RCC_OscInitStruct = {0};
     RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
 
     // Configure the main internal regulator output voltage
     __HAL_RCC_PWR_CLK_ENABLE();
     __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
 
     // Initialize the CPU, AHB and APB bus clocks
     RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
     RCC_OscInitStruct.HSEState = RCC_HSE_ON;
     RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
     RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
     RCC_OscInitStruct.PLL.PLLM = 8;
     RCC_OscInitStruct.PLL.PLLN = 336;
     RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
     RCC_OscInitStruct.PLL.PLLQ = 7;
     
     if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
         Error_Handler();
     }
 
     // Initialize the CPU, AHB and APB bus clocks
     RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                 | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
     RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
     RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
     RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
     RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;
 
     if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK) {
         Error_Handler();
     }
 }
 
 /**
  * @brief Error Handler
  * This function is executed in case of error occurrence
  * @return None
  */
 void Error_Handler(void) {
     // Keep interrupts enabled so the watchdog can reset the MCU.
     // Toggling an LED provides a visible error indicator.
     while (1) {
         HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
         HAL_Delay(200);
     }
 }
 
 /**
  * @brief Initialize system peripherals
  * Sets up GPIOs, communication interfaces, etc.
  * @return None
  */
 void InitializePeripherals(void) {
     // Enable GPIO Clocks
     __HAL_RCC_GPIOA_CLK_ENABLE();
     __HAL_RCC_GPIOB_CLK_ENABLE();
     __HAL_RCC_GPIOC_CLK_ENABLE();
     
     // Configure LED pin
     GPIO_InitTypeDef GPIO_InitStruct = {0};
     GPIO_InitStruct.Pin = GPIO_PIN_5;
     GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
     GPIO_InitStruct.Pull = GPIO_NOPULL;
     GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
     HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
     
     // Initialize boot pin (for bootloader mode detection)
     GPIO_InitStruct.Pin = GPIO_PIN_0;
     GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
     GPIO_InitStruct.Pull = GPIO_PULLUP;
     GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
     HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
 }
 
 /**
  * @brief Main program entry point
  * @return Should never return
  */
 int main(void) {
     // Initialize HAL Library
     HAL_Init();
     
     // Configure system clock
     SystemClock_Config();
     
     // Initialize peripherals
     InitializePeripherals();
     
     // Initialize bootloader
     Bootloader.Initialize();
     
     // Run bootloader main process
     Bootloader.Run();
     
     // Should never reach here as bootloader should either:
     // 1. Jump to application
     // 2. Enter firmware update mode
     // 3. Handle errors
     while (1) {
         // Safety loop in case something goes wrong
         Error_Handler();
     }
 }