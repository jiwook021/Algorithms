/**
  ******************************************************************************
  * @file    stm32f4xx_hal_gpio.h
  * @author  MCD Application Team
  * @brief   Header file of GPIO HAL module.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2017 STMicroelectronics.
  * All rights reserved.
  *
  ******************************************************************************
  */

  #ifndef STM32F4xx_HAL_GPIO_H
  #define STM32F4xx_HAL_GPIO_H
  
  #ifdef __cplusplus
   extern "C" {
  #endif
  
  /* Includes ------------------------------------------------------------------*/
  #include "stm32f4xx_hal_def.h"
  
  /** @addtogroup STM32F4xx_HAL_Driver
    * @{
    */
  
  /** @addtogroup GPIO
    * @{
    */
  
  /* Exported types ------------------------------------------------------------*/
  /** @defgroup GPIO_Exported_Types GPIO Exported Types
    * @{
    */
  
  /**
    * @brief GPIO Init structure definition
    */
  typedef struct
  {
    uint32_t Pin;       /*!< Specifies the GPIO pins to be configured.
                             This parameter can be any value of @ref GPIO_pins_define */
  
    uint32_t Mode;      /*!< Specifies the operating mode for the selected pins.
                             This parameter can be a value of @ref GPIO_mode_define */
  
    uint32_t Pull;      /*!< Specifies the Pull-up or Pull-Down activation for the selected pins.
                             This parameter can be a value of @ref GPIO_pull_define */
  
    uint32_t Speed;     /*!< Specifies the speed for the selected pins.
                             This parameter can be a value of @ref GPIO_speed_define */
  
    uint32_t Alternate;  /*!< Peripheral to be connected to the selected pins.
                              This parameter can be a value of @ref GPIO_Alternate_function_selection */
  }GPIO_InitTypeDef;
  
  /**
    * @brief  GPIO Bit SET and Bit RESET enumeration
    */
  typedef enum
  {
    GPIO_PIN_RESET = 0,
    GPIO_PIN_SET
  }GPIO_PinState;
  /**
    * @}
    */
  
  /* Exported constants --------------------------------------------------------*/
  
  /** @defgroup GPIO_Exported_Constants GPIO Exported Constants
    * @{
    */
  
  /** @defgroup GPIO_pins_define GPIO pins define
    * @{
    */
  #define GPIO_PIN_0                 ((uint16_t)0x0001)  /* Pin 0 selected    */
  #define GPIO_PIN_1                 ((uint16_t)0x0002)  /* Pin 1 selected    */
  #define GPIO_PIN_2                 ((uint16_t)0x0004)  /* Pin 2 selected    */
  #define GPIO_PIN_3                 ((uint16_t)0x0008)  /* Pin 3 selected    */
  #define GPIO_PIN_4                 ((uint16_t)0x0010)  /* Pin 4 selected    */
  #define GPIO_PIN_5                 ((uint16_t)0x0020)  /* Pin 5 selected    */
  #define GPIO_PIN_6                 ((uint16_t)0x0040)  /* Pin 6 selected    */
  #define GPIO_PIN_7                 ((uint16_t)0x0080)  /* Pin 7 selected    */
  #define GPIO_PIN_8                 ((uint16_t)0x0100)  /* Pin 8 selected    */
  #define GPIO_PIN_9                 ((uint16_t)0x0200)  /* Pin 9 selected    */
  #define GPIO_PIN_10                ((uint16_t)0x0400)  /* Pin 10 selected   */
  #define GPIO_PIN_11                ((uint16_t)0x0800)  /* Pin 11 selected   */
  #define GPIO_PIN_12                ((uint16_t)0x1000)  /* Pin 12 selected   */
  #define GPIO_PIN_13                ((uint16_t)0x2000)  /* Pin 13 selected   */
  #define GPIO_PIN_14                ((uint16_t)0x4000)  /* Pin 14 selected   */
  #define GPIO_PIN_15                ((uint16_t)0x8000)  /* Pin 15 selected   */
  #define GPIO_PIN_All               ((uint16_t)0xFFFF)  /* All pins selected */
  /**
    * @}
    */
  
  /** @defgroup GPIO_mode_define GPIO mode define
    * @brief GPIO Configuration Mode
    *        Elements values convention: 0xX0yz00YZ
    *           - X  : GPIO mode or EXTI Mode
    *           - y  : External IT or Event trigger detection
    *           - z  : IO configuration on External IT or Event
    *           - Y  : Output type (Push Pull or Open Drain)
    *           - Z  : IO Direction mode (Input, Output, Alternate or Analog)
    * @{
    */
  #define  GPIO_MODE_INPUT                        0x00000000U   /*!< Input Floating Mode                   */
  #define  GPIO_MODE_OUTPUT_PP                    0x00000001U   /*!< Output Push Pull Mode                 */
  #define  GPIO_MODE_OUTPUT_OD                    0x00000011U   /*!< Output Open Drain Mode                */
  #define  GPIO_MODE_AF_PP                        0x00000002U   /*!< Alternate Function Push Pull Mode     */
  #define  GPIO_MODE_AF_OD                        0x00000012U   /*!< Alternate Function Open Drain Mode    */
  #define  GPIO_MODE_ANALOG                       0x00000003U   /*!< Analog Mode  */
  #define  GPIO_MODE_IT_RISING                    0x10110000U   /*!< External Interrupt Mode with Rising edge trigger detection          */
  #define  GPIO_MODE_IT_FALLING                   0x10210000U   /*!< External Interrupt Mode with Falling edge trigger detection         */
  #define  GPIO_MODE_IT_RISING_FALLING            0x10310000U   /*!< External Interrupt Mode with Rising/Falling edge trigger detection  */
  #define  GPIO_MODE_EVT_RISING                   0x10120000U   /*!< External Event Mode with Rising edge trigger detection               */
  #define  GPIO_MODE_EVT_FALLING                  0x10220000U   /*!< External Event Mode with Falling edge trigger detection              */
  #define  GPIO_MODE_EVT_RISING_FALLING           0x10320000U   /*!< External Event Mode with Rising/Falling edge trigger detection       */
  /**
    * @}
    */
  
  /** @defgroup GPIO_speed_define  GPIO speed define
    * @brief GPIO Output Maximum frequency
    * @{
    */
  #define  GPIO_SPEED_FREQ_LOW              0x00000000U  /*!< Low speed       */
  #define  GPIO_SPEED_FREQ_MEDIUM           0x00000001U  /*!< Medium speed    */
  #define  GPIO_SPEED_FREQ_HIGH             0x00000002U  /*!< High speed      */
  #define  GPIO_SPEED_FREQ_VERY_HIGH        0x00000003U  /*!< Very high speed */
  /**
    * @}
    */
  
   /** @defgroup GPIO_pull_define GPIO pull define
     * @brief GPIO Pull-Up or Pull-Down Activation
     * @{
     */
  #define  GPIO_NOPULL        0x00000000U   /*!< No Pull-up or Pull-down activation  */
  #define  GPIO_PULLUP        0x00000001U   /*!< Pull-up activation                  */
  #define  GPIO_PULLDOWN      0x00000002U   /*!< Pull-down activation                */
  /**
    * @}
    */
  
  /** @defgroup GPIO_Alternate_function_selection GPIO Alternate function selection
    * @brief Alternate function selection
    * @{
    */
  #define GPIO_AF0_RTC_50Hz      ((uint8_t)0x00)  /* RTC_50Hz Alternate Function mapping                       */
  #define GPIO_AF0_MCO           ((uint8_t)0x00)  /* MCO (MCO1 and MCO2) Alternate Function mapping            */
  #define GPIO_AF0_TAMPER        ((uint8_t)0x00)  /* TAMPER (TAMPER_1 and TAMPER_2) Alternate Function mapping */
  #define GPIO_AF0_SWJ           ((uint8_t)0x00)  /* SWJ (SWD and JTAG) Alternate Function mapping             */
  #define GPIO_AF0_TRACE         ((uint8_t)0x00)  /* TRACE Alternate Function mapping                          */
  
  #define GPIO_AF1_TIM1          ((uint8_t)0x01)  /* TIM1 Alternate Function mapping */
  #define GPIO_AF1_TIM2          ((uint8_t)0x01)  /* TIM2 Alternate Function mapping */
  
  #define GPIO_AF2_TIM3          ((uint8_t)0x02)  /* TIM3 Alternate Function mapping */
  #define GPIO_AF2_TIM4          ((uint8_t)0x02)  /* TIM4 Alternate Function mapping */
  #define GPIO_AF2_TIM5          ((uint8_t)0x02)  /* TIM5 Alternate Function mapping */
  
  #define GPIO_AF3_TIM8          ((uint8_t)0x03)  /* TIM8 Alternate Function mapping  */
  #define GPIO_AF3_TIM9          ((uint8_t)0x03)  /* TIM9 Alternate Function mapping  */
  #define GPIO_AF3_TIM10         ((uint8_t)0x03)  /* TIM10 Alternate Function mapping */
  #define GPIO_AF3_TIM11         ((uint8_t)0x03)  /* TIM11 Alternate Function mapping */
  
  #define GPIO_AF4_I2C1          ((uint8_t)0x04)  /* I2C1 Alternate Function mapping */
  #define GPIO_AF4_I2C2          ((uint8_t)0x04)  /* I2C2 Alternate Function mapping */
  #define GPIO_AF4_I2C3          ((uint8_t)0x04)  /* I2C3 Alternate Function mapping */
  
  #define GPIO_AF5_SPI1          ((uint8_t)0x05)  /* SPI1 Alternate Function mapping        */
  #define GPIO_AF5_SPI2          ((uint8_t)0x05)  /* SPI2/I2S2 Alternate Function mapping   */
  #define GPIO_AF5_SPI3          ((uint8_t)0x05)  /* SPI3/I2S3 Alternate Function mapping   */
  #define GPIO_AF5_I2S3ext       ((uint8_t)0x05)  /* I2S3ext_SD Alternate Function mapping  */
  
  #define GPIO_AF6_SPI3          ((uint8_t)0x06)  /* SPI3/I2S3 Alternate Function mapping  */
  #define GPIO_AF6_I2S2ext       ((uint8_t)0x06)  /* I2S2ext_SD Alternate Function mapping */
  
  #define GPIO_AF7_USART1        ((uint8_t)0x07)  /* USART1 Alternate Function mapping     */
  #define GPIO_AF7_USART2        ((uint8_t)0x07)  /* USART2 Alternate Function mapping     */
  #define GPIO_AF7_USART3        ((uint8_t)0x07)  /* USART3 Alternate Function mapping     */
  #define GPIO_AF7_I2S3ext       ((uint8_t)0x07)  /* I2S3ext_SD Alternate Function mapping */
  
  #define GPIO_AF8_UART4         ((uint8_t)0x08)  /* UART4 Alternate Function mapping  */
  #define GPIO_AF8_UART5         ((uint8_t)0x08)  /* UART5 Alternate Function mapping  */
  #define GPIO_AF8_USART6        ((uint8_t)0x08)  /* USART6 Alternate Function mapping */
  
  #define GPIO_AF9_CAN1          ((uint8_t)0x09)  /* CAN1 Alternate Function mapping    */
  #define GPIO_AF9_CAN2          ((uint8_t)0x09)  /* CAN2 Alternate Function mapping    */
  #define GPIO_AF9_TIM12         ((uint8_t)0x09)  /* TIM12 Alternate Function mapping   */
  #define GPIO_AF9_TIM13         ((uint8_t)0x09)  /* TIM13 Alternate Function mapping   */
  #define GPIO_AF9_TIM14         ((uint8_t)0x09)  /* TIM14 Alternate Function mapping   */