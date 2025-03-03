/**
  ******************************************************************************
  * @file    stm32f4xx_hal_flash_ex.h
  * @author  MCD Application Team
  * @brief   Header file of FLASH HAL Extension module.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2017 STMicroelectronics.
  * All rights reserved.
  *
  ******************************************************************************
  */

  #ifndef STM32F4xx_HAL_FLASH_EX_H
  #define STM32F4xx_HAL_FLASH_EX_H
  
  #ifdef __cplusplus
   extern "C" {
  #endif
  
  /* Includes ------------------------------------------------------------------*/
  #include "stm32f4xx_hal_def.h"
  
  /** @addtogroup STM32F4xx_HAL_Driver
    * @{
    */
  
  /** @addtogroup FLASHEx
    * @{
    */
  
  /* Exported types ------------------------------------------------------------*/
  /** @defgroup FLASHEx_Exported_Types FLASH Exported Types
    * @{
    */
  
  /**
    * @brief  FLASH Erase structure definition
    */
  typedef struct
  {
    uint32_t TypeErase;   /*!< TypeErase: Mass erase or sector Erase.
                                This parameter can be a value of @ref FLASHEx_Type_Erase */
  
    uint32_t Banks;       /*!< Banks: Select banks to erase when Mass erase is enabled.
                                This parameter must be a value of @ref FLASHEx_Banks */
  
    uint32_t Sector;      /*!< Sector: Initial FLASH sector to erase when Mass erase is disabled
                                This parameter must be a value of @ref FLASHEx_Sectors */
  
    uint32_t NbSectors;   /*!< NbSectors: Number of sectors to be erased.
                                This parameter must be a value between 1 and (max number of sectors - value of Initial sector)*/
  
    uint32_t VoltageRange;/*!< VoltageRange: The device voltage range which defines the erase parallelism
                                This parameter must be a value of @ref FLASHEx_Voltage_Range */
  
  } FLASH_EraseInitTypeDef;
  
  /**
    * @brief  FLASH Option Bytes Program structure definition
    */
  typedef struct
  {
    uint32_t OptionType;   /*!< OptionType: Option byte to be configured.
                                This parameter can be a value of @ref FLASHEx_Option_Type */
  
    uint32_t WRPState;     /*!< WRPState: Write protection activation or deactivation.
                                This parameter can be a value of @ref FLASHEx_WRP_State */
  
    uint32_t WRPSector;    /*!< WRPSector: specifies the sector(s) to be write protected
                                This parameter can be a value of @ref FLASHEx_Option_Bytes_Write_Protection */
  
    uint32_t Banks;        /*!< Banks: Select banks for WRP activation/deactivation of all sectors
                                This parameter must be a value of @ref FLASHEx_Banks */
  
    uint32_t RDPLevel;     /*!< RDPLevel: Set the read protection level..
                                This parameter can be a value of @ref FLASHEx_Option_Bytes_Read_Protection */
  
    uint32_t BORLevel;     /*!< BORLevel: Set the BOR Level.
                                This parameter can be a value of @ref FLASHEx_BOR_Reset_Level */
  
    uint8_t  USERConfig;   /*!< USERConfig: Program the FLASH User Option Byte:
                                IWDG_SW / RST_STOP / RST_STDBY.
                                This parameter can be a combination of @ref FLASHEx_Option_Bytes_IWatchdog,
                                @ref FLASHEx_Option_Bytes_nRST_STOP and @ref FLASHEx_Option_Bytes_nRST_STDBY */
  } FLASH_OBProgramInitTypeDef;
  
  /**
    * @}
    */
  
  /* Exported constants --------------------------------------------------------*/
  /** @defgroup FLASHEx_Exported_Constants FLASH Exported Constants
    * @{
    */
  
  /** @defgroup FLASHEx_Type_Erase FLASH Type Erase
    * @{
    */
  #define FLASH_TYPEERASE_SECTORS         0x00000000U  /*!< Sectors erase only          */
  #define FLASH_TYPEERASE_MASSERASE       0x00000001U  /*!< Flash Mass erase activation */
  /**
    * @}
    */
  
  /** @defgroup FLASHEx_Voltage_Range FLASH Voltage Range
    * @{
    */
  #define FLASH_VOLTAGE_RANGE_1        0x00000000U  /*!< Flash program/erase by sector operations voltage range: 1.8V to 2.1V */
  #define FLASH_VOLTAGE_RANGE_2        0x00000001U  /*!< Flash program/erase by sector operations voltage range: 2.1V to 2.7V */
  #define FLASH_VOLTAGE_RANGE_3        0x00000002U  /*!< Flash program/erase by sector operations voltage range: 2.7V to 3.6V */
  #define FLASH_VOLTAGE_RANGE_4        0x00000003U  /*!< Flash program/erase by sector operations voltage range: 2.7V to 3.6V + External Vpp */
  /**
    * @}
    */
  
  /** @defgroup FLASHEx_WRP_State FLASH WRP State
    * @{
    */
  #define OB_WRPSTATE_DISABLE       0x00000000U  /*!< Disable the write protection of the desired bank 1 sectors */
  #define OB_WRPSTATE_ENABLE        0x00000001U  /*!< Enable the write protection of the desired bank 1 sectors  */
  /**
    * @}
    */
  
  /** @defgroup FLASHEx_Option_Type FLASH Option Type
    * @{
    */
  #define OPTIONBYTE_WRP        0x00000001U  /*!< WRP option byte configuration  */
  #define OPTIONBYTE_RDP        0x00000002U  /*!< RDP option byte configuration  */
  #define OPTIONBYTE_USER       0x00000004U  /*!< USER option byte configuration */
  #define OPTIONBYTE_BOR        0x00000008U  /*!< BOR option byte configuration  */
  /**
    * @}
    */
  
  /** @defgroup FLASHEx_Option_Bytes_Read_Protection FLASH Option Bytes Read Protection
    * @{
    */
  #define OB_RDP_LEVEL_0   ((uint8_t)0xAA)
  #define OB_RDP_LEVEL_1   ((uint8_t)0x55)
  #define OB_RDP_LEVEL_2   ((uint8_t)0xCC) /*!< Warning: When enabling read protection level 2
                                                it's no more possible to go back to level 1 or 0 */
  /**
    * @}
    */
  
  /** @defgroup FLASHEx_Option_Bytes_IWatchdog FLASH Option Bytes IWatchdog
    * @{
    */
  #define OB_IWDG_SW        ((uint8_t)0x20)  /*!< Software IWDG selected */
  #define OB_IWDG_HW        ((uint8_t)0x00)  /*!< Hardware IWDG selected */
  /**
    * @}
    */
  
  /** @defgroup FLASHEx_Option_Bytes_nRST_STOP FLASH Option Bytes nRST_STOP
    * @{
    */
  #define OB_STOP_NO_RST    ((uint8_t)0x40) /*!< No reset generated when entering in STOP */
  #define OB_STOP_RST       ((uint8_t)0x00) /*!< Reset generated when entering in STOP    */
  /**
    * @}
    */
  
  /** @defgroup FLASHEx_Option_Bytes_nRST_STDBY FLASH Option Bytes nRST_STDBY
    * @{
    */
  #define OB_STDBY_NO_RST   ((uint8_t)0x80) /*!< No reset generated when entering in STANDBY */
  #define OB_STDBY_RST      ((uint8_t)0x00) /*!< Reset generated when entering in STANDBY    */
  /**
    * @}
    */
  
  /** @defgroup FLASHEx_BOR_Reset_Level FLASH BOR Reset Level
    * @{
    */
  #define OB_BOR_LEVEL3     ((uint8_t)0x00)  /*!< Supply voltage ranges from 2.70 to 3.60 V */
  #define OB_BOR_LEVEL2     ((uint8_t)0x04)  /*!< Supply voltage ranges from 2.40 to 2.70 V */
  #define OB_BOR_LEVEL1     ((uint8_t)0x08)  /*!< Supply voltage ranges from 2.10 to 2.40 V */
  #define OB_BOR_OFF        ((uint8_t)0x0C)  /*!< Supply voltage ranges from 1.62 to 2.10 V */
  /**
    * @}
    */
  
  /** @defgroup FLASHEx_Banks FLASH Banks
    * @{
    */
  #define FLASH_BANK_1     1U /*!< Bank 1   */
  #define FLASH_BANK_2     2U /*!< Bank 2   */
  #define FLASH_BANK_BOTH  ((uint32_t)FLASH_BANK_1 | FLASH_BANK_2) /*!< Bank1 and Bank2  */
  /**
    * @}
    */
  
  /** @defgroup FLASHEx_MassErase_bit FLASH Mass Erase bit
    * @{
    */
  #define FLASH_MER_BIT     FLASH_CR_MER  /*!< Only Bank 1 MER bit */
  /**
    * @}
    */
  
  /** @defgroup FLASHEx_Banks_Parallel_mode FLASH Banks parallel mode
    * @{
    */
  #define FLASH_BANK_1_2   ((uint32_t)0x00000000) /*!< Bank 1 and Bank 2 in parallel mode */
  /**
    * @}
    */
  
  /**
    * @}
    */
  
  /* Exported macro ------------------------------------------------------------*/
  /** @defgroup FLASHEx_Exported_Macros FLASH Exported Macros
    * @{
    */
  /**
    * @}
    */
  
  /* Exported functions --------------------------------------------------------*/
  /** @addtogroup FLASHEx_Exported_Functions
    * @{
    */
  
  /** @addtogroup FLASHEx_Exported_Functions_Group1
    * @{
    */
  /* Extension Program operation functions  *************************************/
  HAL_StatusTypeDef HAL_FLASHEx_Erase(FLASH_EraseInitTypeDef *pEraseInit, uint32_t *SectorError);
  HAL_StatusTypeDef HAL_FLASHEx_Erase_IT(FLASH_EraseInitTypeDef *pEraseInit);
  HAL_StatusTypeDef HAL_FLASHEx_OBProgram(FLASH_OBProgramInitTypeDef *pOBInit);
  void              HAL_FLASHEx_OBGetConfig(FLASH_OBProgramInitTypeDef *pOBInit);
  /**
    * @}
    */
  
  /**
    * @}
    */
  /* Private types -------------------------------------------------------------*/
  /* Private variables ---------------------------------------------------------*/
  /* Private constants ---------------------------------------------------------*/
  /* Private macros ------------------------------------------------------------*/
  /** @defgroup FLASHEx_Private_Macros FLASH Private Macros
    * @{
    */
  /**
    * @}
    */
  
  /* Private functions ---------------------------------------------------------*/
  /* Private functions ---------------------------------------------------------*/
  /** @defgroup FLASHEx_Private_Functions FLASH Private Functions
    * @{
    */
  /* Extension Program Operation functions  ************************************/
  void FLASH_Erase_Sector(uint32_t Sector, uint8_t VoltageRange);
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
  
  #endif /* STM32F4xx_HAL_FLASH_EX_H */