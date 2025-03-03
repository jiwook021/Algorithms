/**
  ******************************************************************************
  * @file    stm32f4xx_hal_uart.h
  * @author  MCD Application Team
  * @brief   Header file of UART HAL module.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2017 STMicroelectronics.
  * All rights reserved.
  *
  ******************************************************************************
  */

  #ifndef STM32F4xx_HAL_UART_H
  #define STM32F4xx_HAL_UART_H
  
  #ifdef __cplusplus
   extern "C" {
  #endif
  
  /* Includes ------------------------------------------------------------------*/
  #include "stm32f4xx_hal_def.h"
  
  /** @addtogroup STM32F4xx_HAL_Driver
    * @{
    */
  
  /** @addtogroup UART
    * @{
    */
  
  /* Exported types ------------------------------------------------------------*/
  /** @defgroup UART_Exported_Types UART Exported Types
    * @{
    */
  
  /**
    * @brief UART Init Structure definition
    */
  typedef struct
  {
    uint32_t BaudRate;                  /*!< This member configures the UART communication baud rate.
                                             The baud rate is computed using the following formula:
                                             - IntegerDivider = ((PCLKx) / (8 * (OVR8+1) * (huart->Init.BaudRate)))
                                             - FractionalDivider = ((IntegerDivider - ((uint32_t) IntegerDivider)) * 8 * (OVR8+1)) + 0.5
                                             Where OVR8 is the "oversampling by 8 mode" configuration bit in the CR1 register. */
  
    uint32_t WordLength;                /*!< Specifies the number of data bits transmitted or received in a frame.
                                             This parameter can be a value of @ref UART_Word_Length */
  
    uint32_t StopBits;                  /*!< Specifies the number of stop bits transmitted.
                                             This parameter can be a value of @ref UART_Stop_Bits */
  
    uint32_t Parity;                    /*!< Specifies the parity mode.
                                             This parameter can be a value of @ref UART_Parity
                                             @note When parity is enabled, the computed parity is inserted
                                                   at the MSB position of the transmitted data (9th bit when
                                                   the word length is set to 9 data bits; 8th bit when the
                                                   word length is set to 8 data bits). */
  
    uint32_t Mode;                      /*!< Specifies whether the Receive or Transmit mode is enabled or disabled.
                                             This parameter can be a value of @ref UART_Mode */
  
    uint32_t HwFlowCtl;                 /*!< Specifies whether the hardware flow control mode is enabled or disabled.
                                             This parameter can be a value of @ref UART_Hardware_Flow_Control */
  
    uint32_t OverSampling;              /*!< Specifies whether the Over sampling 8 is enabled or disabled, to achieve higher speed (up to fPCLK/8).
                                             This parameter can be a value of @ref UART_Over_Sampling */
  }UART_InitTypeDef;
  
  /**
    * @brief HAL UART State structures definition
    * @note  HAL UART State value is a combination of 2 different substates: gState and RxState.
    *        - gState contains UART state information related to global Handle management
    *          and also information related to Tx operations.
    *          gState value coding follow below described bitmap :
    *          b7-b6  Error information
    *             00 : No Error
    *             01 : (Not Used)
    *             10 : Timeout
    *             11 : Error
    *          b5     IP initialization status
    *             0  : Reset (IP not initialized)
    *             1  : Init done (IP initialized. HAL UART Init function already called)
    *          b4-b3  (not used)
    *             xx : Should be set to 00
    *          b2     Intrinsic process state
    *             0  : Ready
    *             1  : Busy (IP busy with some configuration or internal operations)
    *          b1     (not used)
    *             x  : Should be set to 0
    *          b0     Tx state
    *             0  : Ready (no Tx operation ongoing)
    *             1  : Busy (Tx operation ongoing)
    *        - RxState contains information related to Rx operations.
    *          RxState value coding follow below described bitmap :
    *          b7-b6  (not used)
    *             xx : Should be set to 00
    *          b5     IP initialization status
    *             0  : Reset (IP not initialized)
    *             1  : Init done (IP initialized)
    *          b4-b2  (not used)
    *            xxx : Should be set to 000
    *          b1     Rx state
    *             0  : Ready (no Rx operation ongoing)
    *             1  : Busy (Rx operation ongoing)
    *          b0     (not used)
    *             x  : Should be set to 0.
    */
  typedef enum
  {
    HAL_UART_STATE_RESET             = 0x00U,    /*!< Peripheral is not yet Initialized
                                                     Value is allowed for gState and RxState */
    HAL_UART_STATE_READY             = 0x20U,    /*!< Peripheral Initialized and ready for use
                                                     Value is allowed for gState and RxState */
    HAL_UART_STATE_BUSY              = 0x24U,    /*!< an internal process is ongoing
                                                     Value is allowed for gState only */
    HAL_UART_STATE_BUSY_TX           = 0x21U,    /*!< Data Transmission process is ongoing
                                                     Value is allowed for gState only */
    HAL_UART_STATE_BUSY_RX           = 0x22U,    /*!< Data Reception process is ongoing
                                                     Value is allowed for RxState only */
    HAL_UART_STATE_BUSY_TX_RX        = 0x23U,    /*!< Data Transmission and Reception process is ongoing
                                                     Not to be used for neither gState nor RxState.
                                                     Value is result of combination (Or) between gState and RxState values */
    HAL_UART_STATE_TIMEOUT           = 0xA0U,    /*!< Timeout state
                                                     Value is allowed for gState only */
    HAL_UART_STATE_ERROR             = 0xE0U     /*!< Error
                                                     Value is allowed for gState only */
  }HAL_UART_StateTypeDef;
  
  /**
    * @brief UART handle Structure definition
    */
  typedef struct
  {
    USART_TypeDef                 *Instance;        /*!< UART registers base address        */
  
    UART_InitTypeDef              Init;             /*!< UART communication parameters      */
  
    uint8_t                       *pTxBuffPtr;      /*!< Pointer to UART Tx transfer Buffer */
  
    uint16_t                      TxXferSize;       /*!< UART Tx Transfer size              */
  
    __IO uint16_t                 TxXferCount;      /*!< UART Tx Transfer Counter           */
  
    uint8_t                       *pRxBuffPtr;      /*!< Pointer to UART Rx transfer Buffer */
  
    uint16_t                      RxXferSize;       /*!< UART Rx Transfer size              */
  
    __IO uint16_t                 RxXferCount;      /*!< UART Rx Transfer Counter           */
  
    DMA_HandleTypeDef             *hdmatx;          /*!< UART Tx DMA Handle parameters      */
  
    DMA_HandleTypeDef             *hdmarx;          /*!< UART Rx DMA Handle parameters      */
  
    HAL_LockTypeDef               Lock;             /*!< Locking object                     */
  
    __IO HAL_UART_StateTypeDef    gState;           /*!< UART state information related to global Handle management
                                                         and also related to Tx operations.
                                                         This parameter can be a value of @ref HAL_UART_StateTypeDef */
  
    __IO HAL_UART_StateTypeDef    RxState;          /*!< UART state information related to Rx operations.
                                                         This parameter can be a value of @ref HAL_UART_StateTypeDef */
  
    __IO uint32_t                 ErrorCode;        /*!< UART Error code                    */
  
  }UART_HandleTypeDef;
  /**
    * @}
    */
  
  /* Exported constants --------------------------------------------------------*/
  /** @defgroup UART_Exported_Constants UART Exported constants
    * @{
    */
  
  /** @defgroup UART_Error_Code UART Error Code
    * @{
    */
  #define HAL_UART_ERROR_NONE         0x00000000U   /*!< No error            */
  #define HAL_UART_ERROR_PE           0x00000001U   /*!< Parity error        */
  #define HAL_UART_ERROR_NE           0x00000002U   /*!< Noise error         */
  #define HAL_UART_ERROR_FE           0x00000004U   /*!< Frame error         */
  #define HAL_UART_ERROR_ORE          0x00000008U   /*!< Overrun error       */
  #define HAL_UART_ERROR_DMA          0x00000010U   /*!< DMA transfer error  */
  /**
    * @}
    */
  
  /** @defgroup UART_Word_Length UART Word Length
    * @{
    */
  #define UART_WORDLENGTH_8B                  0x00000000U
  #define UART_WORDLENGTH_9B                  ((uint32_t)USART_CR1_M)
  /**
    * @}
    */
  
  /** @defgroup UART_Stop_Bits UART Number of Stop Bits
    * @{
    */
  #define UART_STOPBITS_1                     0x00000000U
  #define UART_STOPBITS_2                     ((uint32_t)USART_CR2_STOP_1)
  /**
    * @}
    */
  
  /** @defgroup UART_Parity UART Parity
    * @{
    */
  #define UART_PARITY_NONE                    0x00000000U
  #define UART_PARITY_EVEN                    ((uint32_t)USART_CR1_PCE)
  #define UART_PARITY_ODD                     ((uint32_t)(USART_CR1_PCE | USART_CR1_PS))
  /**
    * @}
    */
  
  /** @defgroup UART_Hardware_Flow_Control UART Hardware Flow Control
    * @{
    */
  #define UART_HWCONTROL_NONE                  0x00000000U
  #define UART_HWCONTROL_RTS                   ((uint32_t)USART_CR3_RTSE)
  #define UART_HWCONTROL_CTS                   ((uint32_t)USART_CR3_CTSE)
  #define UART_HWCONTROL_RTS_CTS               ((uint32_t)(USART_CR3_RTSE | USART_CR3_CTSE))
  /**
    * @}
    */
  
  /** @defgroup UART_Mode UART Transfer Mode
    * @{
    */
  #define UART_MODE_RX                        ((uint32_t)USART_CR1_RE)
  #define UART_MODE_TX                        ((uint32_t)USART_CR1_TE)
  #define UART_MODE_TX_RX                     ((uint32_t)(USART_CR1_TE |USART_CR1_RE))
  /**
    * @}
    */
  
   /** @defgroup UART_State UART State
    * @{
    */
  #define UART_STATE_DISABLE                  0x00000000U
  #define UART_STATE_ENABLE                   ((uint32_t)USART_CR1_UE)
  /**
    * @}
    */
  
  /** @defgroup UART_Over_Sampling UART Over Sampling
    * @{
    */
  #define UART_OVERSAMPLING_16                    0x00000000U
  #define UART_OVERSAMPLING_8                     ((uint32_t)USART_CR1_OVER8)
  /**
    * @}
    */
  
  /** @defgroup UART_LIN_Break_Detection_Length  UART LIN Break Detection Length
    * @{
    */
  #define UART_LINBREAKDETECTLENGTH_10B      0x00000000U
  #define UART_LINBREAKDETECTLENGTH_11B      ((uint32_t)USART_CR2_LBDL)
  /**
    * @}
    */
  
  /** @defgroup UART_WakeUp_functions  UART Wakeup Functions
    * @{
    */
  #define UART_WAKEUPMETHOD_IDLELINE                0x00000000U
  #define UART_WAKEUPMETHOD_ADDRESSMARK             ((uint32_t)USART_CR1_WAKE)
  /**
    * @}
    */
  
  /** @defgroup UART_Flags   UART FLags
    *        Elements values convention: 0xXXXX
    *           - 0xXXXX  : Flag mask in the SR register
    * @{
    */
  #define UART_FLAG_CTS                       ((uint32_t)USART_SR_CTS)
  #define UART_FLAG_LBD                       ((uint32_t)USART_SR_LBD)
  #define UART_FLAG_TXE                       ((uint32_t)USART_SR_TXE)
  #define UART_FLAG_TC                        ((uint32_t)USART_SR_TC)
  #define UART_FLAG_RXNE                      ((uint32_t)USART_SR_RXNE)
  #define UART_FLAG_IDLE                      ((uint32_t)USART_SR_IDLE)
  #define UART_FLAG_ORE                       ((uint32_t)USART_SR_ORE)
  #define UART_FLAG_NE                        ((uint32_t)USART_SR_NE)
  #define UART_FLAG_FE                        ((uint32_t)USART_SR_FE)
  #define UART_FLAG_PE                        ((uint32_t)USART_SR_PE)
  /**
    * @}
    */
  
  /** @defgroup UART_Interrupt_definition  UART Interrupt Definitions
    *        Elements values convention: 0xY000XXXX
    *           - XXXX  : Interrupt mask in the Y register
    *           - Y  : Interrupt source register (2bits)
    *                 - 01: CR1 register
    *                 - 10: CR2 register
    *                 - 11: CR3 register
    * @{
    */
  #define UART_IT_PE                       ((uint32_t)(UART_CR1_REG_INDEX << 28U | USART_CR1_PEIE))
  #define UART_IT_TXE                      ((uint32_t)(UART_CR1_REG_INDEX << 28U | USART_CR1_TXEIE))
  #define UART_IT_TC                       ((uint32_t)(UART_CR1_REG_INDEX << 28U | USART_CR1_TCIE))
  #define UART_IT_RXNE                     ((uint32_t)(UART_CR1_REG_INDEX << 28U | USART_CR1_RXNEIE))
  #define UART_IT_IDLE                     ((uint32_t)(UART_CR1_REG_INDEX << 28U | USART_CR1_IDLEIE))
  
  #define UART_IT_LBD                      ((uint32_t)(UART_CR2_REG_INDEX << 28U | USART_CR2_LBDIE))
  
  #define UART_IT_CTS                      ((uint32_t)(UART_CR3_REG_INDEX << 28U | USART_CR3_CTSIE))
  #define UART_IT_ERR                      ((uint32_t)(UART_CR3_REG_INDEX << 28U | USART_CR3_EIE))
  /**
    * @}
    */
  
  /**
    * @}
    */
  
  /* Exported macro ------------------------------------------------------------*/
  /** @defgroup UART_Exported_Macros UART Exported Macros
    * @{
    */
  
  /** @brief Reset UART handle gstate & RxState
    * @param  __HANDLE__ specifies the UART Handle.
    *         UART Handle selects the USARTx or UARTy peripheral
    *         (USART,UART availability and x,y values depending on device).
    * @retval None
    */
  #define __HAL_UART_RESET_HANDLE_STATE(__HANDLE__)  do{                                                   \
                                                         (__HANDLE__)->gState = HAL_UART_STATE_RESET;      \
                                                         (__HANDLE__)->RxState = HAL_UART_STATE_RESET;     \
                                                       } while(0U)
  
  /** @brief  Flush the UART Data registers
    * @param  __HANDLE__ specifies the UART Handle.
    */
  #define __HAL_UART_FLUSH_DRREGISTER(__HANDLE__) ((__HANDLE__)->Instance->DR)
  
  /** @brief  Check whether the specified UART flag is set or not.
    * @param  __HANDLE__ specifies the UART Handle.
    * @param  __FLAG__ specifies the flag to check.
    *        This parameter can be one of the following values:
    *            @arg UART_FLAG_CTS:  CTS Change flag (not available for UART4 and UART5)
    *            @arg UART_FLAG_LBD:  LIN Break detection flag
    *            @arg UART_FLAG_TXE:  Transmit data register empty flag
    *            @arg UART_FLAG_TC:   Transmission Complete flag
    *            @arg UART_FLAG_RXNE: Receive data register not empty flag
    *            @arg UART_FLAG_IDLE: Idle Line detection flag
    *            @arg UART_FLAG_ORE:  Overrun Error flag
    *            @arg UART_FLAG_NE:   Noise Error flag
    *            @arg UART_FLAG_FE:   Framing Error flag
    *            @arg UART_FLAG_PE:   Parity Error flag
    * @retval The new state of __FLAG__ (TRUE or FALSE).
    */
  #define __HAL_UART_GET_FLAG(__HANDLE__, __FLAG__) (((__HANDLE__)->Instance->SR & (__FLAG__)) == (__FLAG__))
  
  /** @brief  Clear the specified UART pending flag.
    * @param  __HANDLE__ specifies the UART Handle.
    * @param  __FLAG__ specifies the flag to check.
    *          This parameter can be any combination of the following values:
    *            @arg UART_FLAG_CTS:  CTS Change flag (not available for UART4 and UART5).
    *            @arg UART_FLAG_LBD:  LIN Break detection flag.
    *            @arg UART_FLAG_TC:   Transmission Complete flag.
    *            @arg UART_FLAG_RXNE: Receive data register not empty flag.
    *
    * @note   PE (Parity error), FE (Framing error), NE (Noise error), ORE (Overrun
    *          error) and IDLE (Idle line detected) flags are cleared by software
    *          sequence: a read operation to USART_SR register followed by a read
    *          operation to USART_DR register.
    * @note   RXNE flag can be also cleared by a read to the USART_DR register.
    * @note   TC flag can be also cleared by software sequence: a read operation to
    *          USART_SR register followed by a write operation to USART_DR register.
    * @note   TXE flag is cleared only by a write to the USART_DR register.
    *
    * @retval None
    */
  #define __HAL_UART_CLEAR_FLAG(__HANDLE__, __FLAG__) ((__HANDLE__)->Instance->SR = ~(__FLAG__))
  
  /** @brief  Clear the UART PE pending flag.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define __HAL_UART_CLEAR_PEFLAG(__HANDLE__)     \
    do{                                            \
      __IO uint32_t tmpreg = 0x00U;                \
      tmpreg = (__HANDLE__)->Instance->SR;         \
      tmpreg = (__HANDLE__)->Instance->DR;         \
      UNUSED(tmpreg);                             \
    } while(0U)
  
  /** @brief  Clear the UART FE pending flag.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define __HAL_UART_CLEAR_FEFLAG(__HANDLE__) __HAL_UART_CLEAR_PEFLAG(__HANDLE__)
  
  /** @brief  Clear the UART NE pending flag.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define __HAL_UART_CLEAR_NEFLAG(__HANDLE__) __HAL_UART_CLEAR_PEFLAG(__HANDLE__)
  
  /** @brief  Clear the UART ORE pending flag.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define __HAL_UART_CLEAR_OREFLAG(__HANDLE__) __HAL_UART_CLEAR_PEFLAG(__HANDLE__)
  
  /** @brief  Clear the UART IDLE pending flag.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define __HAL_UART_CLEAR_IDLEFLAG(__HANDLE__) __HAL_UART_CLEAR_PEFLAG(__HANDLE__)
  
  /** @brief  Enable the specified UART interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @param  __INTERRUPT__ specifies the UART interrupt source to enable.
    *          This parameter can be one of the following values:
    *            @arg UART_IT_CTS:  CTS change interrupt
    *            @arg UART_IT_LBD:  LIN Break detection interrupt
    *            @arg UART_IT_TXE:  Transmit Data Register empty interrupt
    *            @arg UART_IT_TC:   Transmission complete interrupt
    *            @arg UART_IT_RXNE: Receive Data register not empty interrupt
    *            @arg UART_IT_IDLE: Idle line detection interrupt
    *            @arg UART_IT_PE:   Parity Error interrupt
    *            @arg UART_IT_ERR:  Error interrupt(Frame error, noise error, overrun error)
    * @retval None
    */
  #define __HAL_UART_ENABLE_IT(__HANDLE__, __INTERRUPT__)   ((((__INTERRUPT__) >> 28U) == UART_CR1_REG_INDEX)? ((__HANDLE__)->Instance->CR1 |= ((__INTERRUPT__) & UART_IT_MASK)): \
                                                              (((__INTERRUPT__) >> 28U) == UART_CR2_REG_INDEX)? ((__HANDLE__)->Instance->CR2 |= ((__INTERRUPT__) & UART_IT_MASK)): \
                                                              ((__HANDLE__)->Instance->CR3 |= ((__INTERRUPT__) & UART_IT_MASK)))
  
  /** @brief  Disable the specified UART interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @param  __INTERRUPT__ specifies the UART interrupt source to disable.
    *          This parameter can be one of the following values:
    *            @arg UART_IT_CTS:  CTS change interrupt
    *            @arg UART_IT_LBD:  LIN Break detection interrupt
    *            @arg UART_IT_TXE:  Transmit Data Register empty interrupt
    *            @arg UART_IT_TC:   Transmission complete interrupt
    *            @arg UART_IT_RXNE: Receive Data register not empty interrupt
    *            @arg UART_IT_IDLE: Idle line detection interrupt
    *            @arg UART_IT_PE:   Parity Error interrupt
    *            @arg UART_IT_ERR:  Error interrupt(Frame error, noise error, overrun error)
    * @retval None
    */
  #define __HAL_UART_DISABLE_IT(__HANDLE__, __INTERRUPT__)  ((((__INTERRUPT__) >> 28U) == UART_CR1_REG_INDEX)? ((__HANDLE__)->Instance->CR1 &= ~((__INTERRUPT__) & UART_IT_MASK)): \
                                                              (((__INTERRUPT__) >> 28U) == UART_CR2_REG_INDEX)? ((__HANDLE__)->Instance->CR2 &= ~((__INTERRUPT__) & UART_IT_MASK)): \
                                                              ((__HANDLE__)->Instance->CR3 &= ~ ((__INTERRUPT__) & UART_IT_MASK)))
  
  /** @brief  Check whether the specified UART interrupt has occurred or not.
    * @param  __HANDLE__ specifies the UART Handle.
    * @param  __IT__ specifies the UART interrupt to check.
    *          This parameter can be one of the following values:
    *            @arg UART_IT_CTS: CTS change interrupt (not available for UART4 and UART5)
    *            @arg UART_IT_LBD: LIN Break detection interrupt
    *            @arg UART_IT_TXE: Transmit Data Register empty interrupt
    *            @arg UART_IT_TC:  Transmission complete interrupt
    *            @arg UART_IT_RXNE: Receive Data register not empty interrupt
    *            @arg UART_IT_IDLE: Idle line detection interrupt
    *            @arg UART_IT_ERR: Error interrupt
    *            @arg UART_IT_PE: Parity Error interrupt
    * @retval The new state of __IT__ (TRUE or FALSE).
    */
  #define __HAL_UART_GET_IT_SOURCE(__HANDLE__, __IT__) (((((__IT__) >> 28U) == UART_CR1_REG_INDEX)? (__HANDLE__)->Instance->CR1:(((((uint32_t)(__IT__)) >> 28U) == UART_CR2_REG_INDEX)? \
                                                        (__HANDLE__)->Instance->CR2 : (__HANDLE__)->Instance->CR3)) & (((uint32_t)(__IT__)) & UART_IT_MASK))
  
  /** @brief  Enable UART
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define __HAL_UART_ENABLE(__HANDLE__)               ((__HANDLE__)->Instance->CR1 |=  USART_CR1_UE)
  
  /** @brief  Disable UART
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define __HAL_UART_DISABLE(__HANDLE__)              ((__HANDLE__)->Instance->CR1 &=  ~USART_CR1_UE)
  /**
    * @}
    */
  
  /* Exported functions --------------------------------------------------------*/
  /** @addtogroup UART_Exported_Functions
    * @{
    */
  
  /** @addtogroup UART_Exported_Functions_Group1
    * @{
    */
  /* Initialization/de-initialization functions  **********************************/
  HAL_StatusTypeDef HAL_UART_Init(UART_HandleTypeDef *huart);
  HAL_StatusTypeDef HAL_HalfDuplex_Init(UART_HandleTypeDef *huart);
  HAL_StatusTypeDef HAL_LIN_Init(UART_HandleTypeDef *huart, uint32_t BreakDetectLength);
  HAL_StatusTypeDef HAL_MultiProcessor_Init(UART_HandleTypeDef *huart, uint8_t Address, uint32_t WakeUpMethod);
  HAL_StatusTypeDef HAL_UART_DeInit (UART_HandleTypeDef *huart);
  void HAL_UART_MspInit(UART_HandleTypeDef *huart);
  void HAL_UART_MspDeInit(UART_HandleTypeDef *huart);
  /**
    * @}
    */
  
  /** @addtogroup UART_Exported_Functions_Group2
    * @{
    */
  /* IO operation functions *******************************************************/
  HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *huart, uint8_t *pData, uint16_t Size, uint32_t Timeout);
  HAL_StatusTypeDef HAL_UART_Receive(UART_HandleTypeDef *huart, uint8_t *pData, uint16_t Size, uint32_t Timeout);
  HAL_StatusTypeDef HAL_UART_Transmit_IT(UART_HandleTypeDef *huart, uint8_t *pData, uint16_t Size);
  HAL_StatusTypeDef HAL_UART_Receive_IT(UART_HandleTypeDef *huart, uint8_t *pData, uint16_t Size);
  HAL_StatusTypeDef HAL_UART_Transmit_DMA(UART_HandleTypeDef *huart, uint8_t *pData, uint16_t Size);
  HAL_StatusTypeDef HAL_UART_Receive_DMA(UART_HandleTypeDef *huart, uint8_t *pData, uint16_t Size);
  HAL_StatusTypeDef HAL_UART_DMAPause(UART_HandleTypeDef *huart);
  HAL_StatusTypeDef HAL_UART_DMAResume(UART_HandleTypeDef *huart);
  HAL_StatusTypeDef HAL_UART_DMAStop(UART_HandleTypeDef *huart);
  /* Transfer Abort functions */
  HAL_StatusTypeDef HAL_UART_Abort(UART_HandleTypeDef *huart);
  HAL_StatusTypeDef HAL_UART_AbortTransmit(UART_HandleTypeDef *huart);
  HAL_StatusTypeDef HAL_UART_AbortReceive(UART_HandleTypeDef *huart);
  HAL_StatusTypeDef HAL_UART_Abort_IT(UART_HandleTypeDef *huart);
  HAL_StatusTypeDef HAL_UART_AbortTransmit_IT(UART_HandleTypeDef *huart);
  HAL_StatusTypeDef HAL_UART_AbortReceive_IT(UART_HandleTypeDef *huart);
  
  void HAL_UART_IRQHandler(UART_HandleTypeDef *huart);
  void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart);
  void HAL_UART_TxHalfCpltCallback(UART_HandleTypeDef *huart);
  void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart);
  void HAL_UART_RxHalfCpltCallback(UART_HandleTypeDef *huart);
  void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart);
  void HAL_UART_AbortCpltCallback (UART_HandleTypeDef *huart);
  void HAL_UART_AbortTransmitCpltCallback (UART_HandleTypeDef *huart);
  void HAL_UART_AbortReceiveCpltCallback (UART_HandleTypeDef *huart);
  /**
    * @}
    */
  
  /** @addtogroup UART_Exported_Functions_Group3
    * @{
    */
  /* Peripheral Control functions  ************************************************/
  HAL_StatusTypeDef HAL_LIN_SendBreak(UART_HandleTypeDef *huart);
  HAL_StatusTypeDef HAL_MultiProcessor_EnterMuteMode(UART_HandleTypeDef *huart);
  HAL_StatusTypeDef HAL_MultiProcessor_ExitMuteMode(UART_HandleTypeDef *huart);
  HAL_StatusTypeDef HAL_HalfDuplex_EnableTransmitter(UART_HandleTypeDef *huart);
  HAL_StatusTypeDef HAL_HalfDuplex_EnableReceiver(UART_HandleTypeDef *huart);
  /**
    * @}
    */
  
  /** @addtogroup UART_Exported_Functions_Group4
    * @{
    */
  /* Peripheral State functions  **************************************************/
  HAL_UART_StateTypeDef HAL_UART_GetState(UART_HandleTypeDef *huart);
  uint32_t HAL_UART_GetError(UART_HandleTypeDef *huart);
  /**
    * @}
    */
  
  /**
    * @}
    */
  /* Private types -------------------------------------------------------------*/
  /* Private variables ---------------------------------------------------------*/
  /* Private constants ---------------------------------------------------------*/
  /** @defgroup UART_Private_Constants UART Private Constants
    * @{
    */
  /** @brief UART interruptions flag mask
    *
    */
  #define UART_IT_MASK                     0x0000FFFFU
  
  /** @brief UART CR1 register index
    *
    */
  #define UART_CR1_REG_INDEX               1U
  
  /** @brief UART CR2 register index
    *
    */
  #define UART_CR2_REG_INDEX               2U
  
  /** @brief UART CR3 register index
    *
    */
  #define UART_CR3_REG_INDEX               3U
  
  /**
    * @}
    */
  
  /* Private macros ------------------------------------------------------------*/
  /** @defgroup UART_Private_Macros UART Private Macros
    * @{
    */
  /** @brief  BRR division operation to set BRR register with OVER8=0
    * @param  __PCLK__ UART clock
    * @param  __BAUD__ Baud rate set by the user
    * @retval Division result
    */
  #define UART_DIV_SAMPLING16(__PCLK__, __BAUD__)   (((__PCLK__) + ((__BAUD__)/2U)) / (__BAUD__))
  
  /** @brief  BRR division operation to set BRR register with OVER8=1
    * @param  __PCLK__ UART clock
    * @param  __BAUD__ Baud rate set by the user
    * @retval Division result
    */
  #define UART_DIV_SAMPLING8(__PCLK__, __BAUD__)    ((((__PCLK__)*2U) + ((__BAUD__)/2U)) / (__BAUD__))
  
  /** @brief  Check whether the specified UART flag is set or not.
    * @param  __SR__ UART SR register
    * @param  __FLAG__ specifies the flag to check.
    * @retval The new state of __FLAG__ (TRUE or FALSE).
    */
  #define UART_FLAG_ISSET(__SR__, __FLAG__)         (((__SR__) & (__FLAG__)) == (__FLAG__))
  
  /** @brief  Clear the specified UART pending flag.
    * @param  __FLAG__ specifies the flag to check.
    * @retval None
    */
  #define UART_CLEAR_FLAG(__FLAG__)                 ((__FLAG__) = ~(__FLAG__))
  
  /** @brief  Clear the UART pending flags which are cleared by writing 1 and 0.
    * @param  __HANDLE__ specifies the UART Handle.
    * @param  __FLAG__ specifies the UART flags to clear.
    * @retval None
    */
  #define UART_CLEAR_PEFLAG(__HANDLE__, __FLAG__)   do{(__HANDLE__)->Instance->SR = ~(__FLAG__); \
                                                       (__HANDLE__)->Instance->DR;\
                                                      }while(0U)
  #define UART_MASK_FLAG(__FLAG__)                  ((__FLAG__) & 0x0000FFFFU)
  
  /** @brief  Checks whether the specified UART flag is set or not.
    * @param  __HANDLE__ specifies the UART Handle.
    * @param  __FLAG__ specifies the flag to check.
    * @retval The new state of __FLAG__ (TRUE or FALSE).
    */
  #define UART_CHECK_FLAG(__HANDLE__, __FLAG__)     (((__HANDLE__)->Instance->SR & ((__FLAG__) & 0x0000FFFFU)) == ((__FLAG__) & 0x0000FFFFU))
  
  /** @brief  Check whether the specified UART interrupt has occurred or not.
    * @param  __HANDLE__ specifies the UART Handle.
    * @param  __IT__ specifies the UART interrupt source to check.
    * @retval The new state of __IT__ (TRUE or FALSE).
    */
  #define UART_CHECK_IT(__HANDLE__, __IT__)         (UART_CHECK_FLAG((__HANDLE__), (__IT__)) \
                                                      && UART_CHECK_IT_SOURCE((__HANDLE__), (__IT__)))
  
  /** @brief  Check whether the specified UART interrupt source is enabled or not.
    * @param  __HANDLE__ specifies the UART Handle.
    * @param  __IT__ specifies the UART interrupt source to check.
    * @retval The new state of __IT__ (TRUE or FALSE).
    */
  #define UART_CHECK_IT_SOURCE(__HANDLE__, __IT__)  ((((((((uint8_t)(__IT__)) >> 5U) == 1U)? (__HANDLE__)->Instance->CR1:(((((uint8_t)(__IT__)) >> 5U) == 2U)? \
                                                        (__HANDLE__)->Instance->CR2 : (__HANDLE__)->Instance->CR3)) & (1U << (((uint16_t)(__IT__)) & UART_IT_MASK)))!= 0U))
  
  /** @brief  Macro to enable the UART transmit data register empty interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_ENABLE_IT_TXE(__HANDLE__)   ((__HANDLE__)->Instance->CR1 |= USART_CR1_TXEIE)
  
  
  /** @brief  Macro to disable the UART transmit data register empty interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_DISABLE_IT_TXE(__HANDLE__)  ((__HANDLE__)->Instance->CR1 &= ~USART_CR1_TXEIE)
  
  /** @brief  Macro to enable the UART parity error interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_ENABLE_IT_PE(__HANDLE__)    ((__HANDLE__)->Instance->CR1 |= USART_CR1_PEIE)
  
  /** @brief  Macro to disable the UART parity error interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_DISABLE_IT_PE(__HANDLE__)   ((__HANDLE__)->Instance->CR1 &= ~USART_CR1_PEIE)
  
  /** @brief  Macro to enable the UART frame error interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_ENABLE_IT_ERR(__HANDLE__)   ((__HANDLE__)->Instance->CR3 |= USART_CR3_EIE)
  
  /** @brief  Macro to disable the UART frame error interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_DISABLE_IT_ERR(__HANDLE__)  ((__HANDLE__)->Instance->CR3 &= ~USART_CR3_EIE)
  
  /** @brief  Macro to enable the UART receive data register not empty interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_ENABLE_IT_RXNE(__HANDLE__)  ((__HANDLE__)->Instance->CR1 |= USART_CR1_RXNEIE)
  
  /** @brief  Macro to disable the UART receive data register not empty interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_DISABLE_IT_RXNE(__HANDLE__) ((__HANDLE__)->Instance->CR1 &= ~USART_CR1_RXNEIE)
  
  /** @brief  Macro to enable the UART break detection interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_ENABLE_IT_LBD(__HANDLE__)   ((__HANDLE__)->Instance->CR2 |= USART_CR2_LBDIE)
  
  /** @brief  Macro to disable the UART break detection interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_DISABLE_IT_LBD(__HANDLE__)  ((__HANDLE__)->Instance->CR2 &= ~USART_CR2_LBDIE)
  
  /** @brief  Macro to enable the UART transmit complete interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_ENABLE_IT_TC(__HANDLE__)    ((__HANDLE__)->Instance->CR1 |= USART_CR1_TCIE)
  
  /** @brief  Macro to disable the UART transmit complete interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_DISABLE_IT_TC(__HANDLE__)   ((__HANDLE__)->Instance->CR1 &= ~USART_CR1_TCIE)
  
  /** @brief  Macro to enable the UART CTS interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_ENABLE_IT_CTS(__HANDLE__)   ((__HANDLE__)->Instance->CR3 |= USART_CR3_CTSIE)
  
  /** @brief  Macro to disable the UART CTS interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_DISABLE_IT_CTS(__HANDLE__)  ((__HANDLE__)->Instance->CR3 &= ~USART_CR3_CTSIE)
  
  /** @brief  Macro to enable the UART idle interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_ENABLE_IT_IDLE(__HANDLE__)  ((__HANDLE__)->Instance->CR1 |= USART_CR1_IDLEIE)
  
  /** @brief  Macro to disable the UART idle interrupt.
    * @param  __HANDLE__ specifies the UART Handle.
    * @retval None
    */
  #define UART_DISABLE_IT_IDLE(__HANDLE__) ((__HANDLE__)->Instance->CR1 &= ~USART_CR1_IDLEIE)
  
  /** @brief  Macro to check whether the specified UART interrupt is set.
    * @param  __HANDLE__ specifies the UART Handle.
    * @param  __FLAG__ specifies the flag to check.
    * @retval None
    */
  #define UART_CHECK_IT(__HANDLE__, __FLAG__) ((((__HANDLE__)->Instance->SR & (__FLAG__)) == (__FLAG__)) ? SET : RESET)
  
  /** @brief  Check whether the specified UART interrupt has occurred or not.
    * @param  __HANDLE__ specifies the UART Handle.
    * @param  __FLAG__ specifies the flag to check.
    * @param  __SOURCE__ specifies the interrupt source bit to check.
    * @retval The new state of __FLAG__ (TRUE or FALSE).
    */
  #define UART_CHECK_IT_SOURCE(__HANDLE__, __FLAG__, __SOURCE__) \
    (UART_CHECK_FLAG((__HANDLE__), (__FLAG__)) && UART_CHECK_IT_SOURCE((__HANDLE__), (__SOURCE__)))
  
  /**
    * @}
    */
  
  /* Private functions ---------------------------------------------------------*/
  /** @defgroup UART_Private_Functions UART Private Functions
    * @{
    */
  
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
  
  #endif /* STM32F4xx_HAL_UART_H */