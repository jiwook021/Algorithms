/**
  ******************************************************************************
  * @file    stm32f4xx_hal.h
  * @author  MCD Application Team
  * @brief   This file contains all the functions prototypes for the HAL
  *          module driver.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2017 STMicroelectronics.
  * All rights reserved.
  *
  ******************************************************************************
  */

  #ifndef STM32F4xx_HAL_H
  #define STM32F4xx_HAL_H
  
  #ifdef __cplusplus
   extern "C" {
  #endif
  
  /* Includes ------------------------------------------------------------------*/
  #include "stm32f4xx_hal_conf.h"
  #include "stm32f4xx_hal_def.h"
  
  /* Exported types ------------------------------------------------------------*/
  /* Exported constants --------------------------------------------------------*/
  /** @defgroup HAL_Exported_Constants HAL Exported Constants
    * @{
    */
  
  /** @defgroup HAL_TICK_FREQ Tick Frequency
    * @{
    */
  typedef enum
  {
    HAL_TICK_FREQ_10HZ         = 100U,
    HAL_TICK_FREQ_100HZ        = 10U,
    HAL_TICK_FREQ_1KHZ         = 1U,
    HAL_TICK_FREQ_DEFAULT      = HAL_TICK_FREQ_1KHZ
  } HAL_TickFreqTypeDef;
  /**
    * @}
    */
  
  /**
    * @}
    */
  
  /* Exported macro ------------------------------------------------------------*/
  /** @defgroup HAL_Exported_Macros HAL Exported Macros
    * @{
    */
  
  /** @brief  Freeze/Unfreeze Peripherals in Debug mode
    */
  #define __HAL_DBGMCU_FREEZE_TIM2()           (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_TIM2_STOP))
  #define __HAL_DBGMCU_FREEZE_TIM3()           (DBGMCU->APB1FZ |= (DBGMCU_APB1_FZ_DBG_TIM3_STOP))
  /* ... more debug macros ... */
  
  /** @brief  Enable the specified peripheral clock.
    * @note   After reset, the peripheral clock (used for registers read/write access)
    *         is disabled and the application software has to enable this clock before
    *         using it.
    */
  #define __HAL_RCC_GPIOA_CLK_ENABLE()   do { \
                                          __IO uint32_t tmpreg = 0x00U; \
                                          SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOAEN);\
                                          /* Delay after an RCC peripheral clock enabling */ \
                                          tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOAEN);\
                                          UNUSED(tmpreg); \
                                        } while(0U)
  #define __HAL_RCC_GPIOB_CLK_ENABLE()   do { \
                                          __IO uint32_t tmpreg = 0x00U; \
                                          SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOBEN);\
                                          /* Delay after an RCC peripheral clock enabling */ \
                                          tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOBEN);\
                                          UNUSED(tmpreg); \
                                        } while(0U)
  #define __HAL_RCC_GPIOC_CLK_ENABLE()   do { \
                                          __IO uint32_t tmpreg = 0x00U; \
                                          SET_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOCEN);\
                                          /* Delay after an RCC peripheral clock enabling */ \
                                          tmpreg = READ_BIT(RCC->AHB1ENR, RCC_AHB1ENR_GPIOCEN);\
                                          UNUSED(tmpreg); \
                                        } while(0U)
  
  #define __HAL_RCC_USART2_CLK_ENABLE()   do { \
                                          __IO uint32_t tmpreg = 0x00U; \
                                          SET_BIT(RCC->APB1ENR, RCC_APB1ENR_USART2EN);\
                                          /* Delay after an RCC peripheral clock enabling */ \
                                          tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_USART2EN);\
                                          UNUSED(tmpreg); \
                                        } while(0U)
  
  #define __HAL_RCC_PWR_CLK_ENABLE()   do { \
                                         __IO uint32_t tmpreg = 0x00U; \
                                         SET_BIT(RCC->APB1ENR, RCC_APB1ENR_PWREN);\
                                         /* Delay after an RCC peripheral clock enabling */ \
                                         tmpreg = READ_BIT(RCC->APB1ENR, RCC_APB1ENR_PWREN);\
                                         UNUSED(tmpreg); \
                                       } while(0U)
  
  /* Flash memory control */
  #define __HAL_FLASH_CLEAR_FLAG(__FLAG__)   ((FLASH->SR) = (__FLAG__))
  
  /* PWR voltage scaling */
  #define __HAL_PWR_VOLTAGESCALING_CONFIG(__REGULATOR__) (MODIFY_REG(PWR->CR, PWR_CR_VOS, (__REGULATOR__)))
  
  /**
    * @}
    */
  
  /* Exported functions --------------------------------------------------------*/
  /** @addtogroup HAL_Exported_Functions
    * @{
    */
  /** @addtogroup HAL_Exported_Functions_Group1
    * @{
    */
  HAL_StatusTypeDef HAL_Init(void);
  HAL_StatusTypeDef HAL_DeInit(void);
  void HAL_MspInit(void);
  void HAL_MspDeInit(void);
  HAL_StatusTypeDef HAL_InitTick(uint32_t TickPriority);
  void HAL_IncTick(void);
  void HAL_Delay(uint32_t Delay);
  uint32_t HAL_GetTick(void);
  uint32_t HAL_GetTickPrio(void);
  HAL_StatusTypeDef HAL_SetTickFreq(HAL_TickFreqTypeDef Freq);
  HAL_TickFreqTypeDef HAL_GetTickFreq(void);
  void HAL_SuspendTick(void);
  void HAL_ResumeTick(void);
  void HAL_NVIC_SystemReset(void);
  uint32_t HAL_SYSTICK_Config(uint32_t TicksNumb);
  /**
    * @}
    */
  
  /**
    * @}
    */
  
  /**
    * @}
    */
  
  #ifdef __cplusplus
  }
  #endif
  
  #endif /* STM32F4xx_HAL_H */