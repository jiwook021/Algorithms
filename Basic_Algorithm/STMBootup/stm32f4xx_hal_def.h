/**
  ******************************************************************************
  * @file    stm32f4xx_hal_def.h
  * @author  MCD Application Team
  * @brief   This file contains HAL common defines, enumeration, macros and
  *          structures definitions.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2017 STMicroelectronics.
  * All rights reserved.
  *
  ******************************************************************************
  */

  #ifndef STM32F4xx_HAL_DEF_H
  #define STM32F4xx_HAL_DEF_H
  
  #ifdef __cplusplus
   extern "C" {
  #endif
  
  /* Includes ------------------------------------------------------------------*/
  #include "stm32f4xx.h"
  #include <stddef.h>
  
  /* Exported types ------------------------------------------------------------*/
  
  /**
    * @brief  HAL Status structures definition
    */
  typedef enum
  {
    HAL_OK       = 0x00U,
    HAL_ERROR    = 0x01U,
    HAL_BUSY     = 0x02U,
    HAL_TIMEOUT  = 0x03U
  } HAL_StatusTypeDef;
  
  /**
    * @brief  HAL Lock structures definition
    */
  typedef enum
  {
    HAL_UNLOCKED = 0x00U,
    HAL_LOCKED   = 0x01U
  } HAL_LockTypeDef;
  
  /* Exported macro ------------------------------------------------------------*/
  #define HAL_MAX_DELAY      0xFFFFFFFFU
  
  #define HAL_IS_BIT_SET(REG, BIT)         (((REG) & (BIT)) != 0U)
  #define HAL_IS_BIT_CLR(REG, BIT)         (((REG) & (BIT)) == 0U)
  
  #define __HAL_LINKDMA(__HANDLE__, __PPP_DMA_FIELD__, __DMA_HANDLE__)               \
                          do{                                                      \
                                (__HANDLE__)->__PPP_DMA_FIELD__ = &(__DMA_HANDLE__); \
                                (__DMA_HANDLE__).Parent = (__HANDLE__);             \
                            } while(0U)
  
  #define UNUSED(X) (void)X      /* To avoid gcc/g++ warnings */
  
  /** @brief Reset the Handle's State field.
    * @param __HANDLE__ specifies the Peripheral Handle.
    * @note  This macro can be used for the following purpose:
    *          - When the Handle is declared as local variable; before passing it to HAL_PPP_Init();
    *          - To get or return function parameters when the relevant parameter is a Handle.
    *          - When the Handle is declared as a global constant; at initialization.
    * @note  Devices having more than 32 peripherals must check the state outside this macro
    *        (e.g. using HAL_HANDLE_STATE_GET)
    * @retval None
    */
  #define __HAL_RESET_HANDLE_STATE(__HANDLE__) ((__HANDLE__)->State = 0U)
  
  #if (USE_RTOS == 1U)
    /* Reserved for future use */
    #error "USE_RTOS should be 0 in the current HAL release"
  #else
    #define __HAL_LOCK(__HANDLE__)                                           \
                                  do{                                        \
                                      if((__HANDLE__)->Lock == HAL_LOCKED)   \
                                      {                                      \
                                         return HAL_BUSY;                    \
                                      }                                      \
                                      else                                   \
                                      {                                      \
                                         (__HANDLE__)->Lock = HAL_LOCKED;    \
                                      }                                      \
                                    }while (0U)
  
    #define __HAL_UNLOCK(__HANDLE__)                                          \
                                    do{                                       \
                                        (__HANDLE__)->Lock = HAL_UNLOCKED;    \
                                      }while (0U)
  #endif /* USE_RTOS */
  
  #if defined (__CC_ARM)
  #pragma diag_suppress 3731
  #endif
  
  /** @brief  Set a bit in the specified register.
    * @param  REG specifies the register to set.
    * @param  BIT specifies the bit to set. This parameter can be one of GPIO_PinTypeDef values.
    * @retval None
    */
  #define SET_BIT(REG, BIT)     ((REG) |= (BIT))
  
  /** @brief  Clear a bit in the specified register.
    * @param  REG specifies the register to clear.
    * @param  BIT specifies the bit to be cleared.
    * @retval None
    */
  #define CLEAR_BIT(REG, BIT)   ((REG) &= ~(BIT))
  
  /** @brief  Read a value from a register.
    * @param  REG specifies the register to read.
    * @retval The value of the read register.
    */
  #define READ_REG(REG)         ((REG))
  
  /** @brief  Write a value in a register.
    * @param  REG specifies the register to be written.
    * @param  VAL specifies the value to be written in the register.
    * @retval None
    */
  #define WRITE_REG(REG, VAL)   ((REG) = (VAL))
  
  /** @brief  Modify a register.
    * @param  REG specifies the register to be modified.
    * @param  CLEARMASK specifies the bits field to be cleared.
    * @param  SETMASK specifies the bits field to be set.
    * @retval None
    */
  #define MODIFY_REG(REG, CLEARMASK, SETMASK)  WRITE_REG((REG), (((READ_REG(REG)) & (~(CLEARMASK))) | (SETMASK)))
  
  /* Exported functions --------------------------------------------------------*/
  #define NVIC_PRIORITYGROUP_0         0x00000007U /*!< 0 bits for pre-emption priority
                                                        4 bits for subpriority */
  #define NVIC_PRIORITYGROUP_1         0x00000006U /*!< 1 bits for pre-emption priority
                                                        3 bits for subpriority */
  #define NVIC_PRIORITYGROUP_2         0x00000005U /*!< 2 bits for pre-emption priority
                                                        2 bits for subpriority */
  #define NVIC_PRIORITYGROUP_3         0x00000004U /*!< 3 bits for pre-emption priority
                                                        1 bits for subpriority */
  #define NVIC_PRIORITYGROUP_4         0x00000003U /*!< 4 bits for pre-emption priority
                                                        0 bits for subpriority */
  
  /* Initialization and de-initialization functions *****************************/
  HAL_StatusTypeDef HAL_Init(void);
  HAL_StatusTypeDef HAL_DeInit(void);
  void HAL_MspInit(void);
  void HAL_MspDeInit(void);
  HAL_StatusTypeDef HAL_InitTick (uint32_t TickPriority);
  
  /* This is the configuration section for CPU ID, Revision ID */
  #define __HAL_DBGMCU_GetREVID() ((DBGMCU->IDCODE) >> 16U)
  #define __HAL_DBGMCU_GetDEVID() ((DBGMCU->IDCODE) & IDCODE_DEVID_MASK)
  
  #ifdef __cplusplus
  }
  #endif
  
  #endif /* STM32F4xx_HAL_DEF_H */