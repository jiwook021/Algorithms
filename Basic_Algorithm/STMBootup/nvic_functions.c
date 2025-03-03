/**
  ******************************************************************************
  * @file    nvic_functions.c
  * @author  Claude
  * @brief   NVIC (Nested Vectored Interrupt Controller) functions
  ******************************************************************************
  */

  #include "stm32f4xx_hal.h"

  /**
   * @brief Sets the priority grouping field using the required unlocking sequence.
   * @param PriorityGroup - Priority grouping field
   * @note   When the NVIC_PriorityGroup_0 is selected, IRQ preemption is no more possible. 
   *         The pending IRQ priority will be managed only by the subpriority. 
   */
  void HAL_NVIC_SetPriorityGrouping(uint32_t PriorityGroup)
  {
    /* Set the PRIGROUP[10:8] bits according to the PriorityGroup parameter value */
    NVIC_SetPriorityGrouping(PriorityGroup);
  }
  
  /**
   * @brief Sets the priority of an interrupt.
   * @param IRQn - External interrupt number
   * @param PreemptPriority - Preemption priority
   * @param SubPriority - Sub-priority
   */
  void HAL_NVIC_SetPriority(IRQn_Type IRQn, uint32_t PreemptPriority, uint32_t SubPriority)
  {
    uint32_t prioritygroup = 0x00U;
    
    /* Check the parameters */
    assert_param(IS_NVIC_SUB_PRIORITY(SubPriority));
    assert_param(IS_NVIC_PREEMPTION_PRIORITY(PreemptPriority));
    
    prioritygroup = NVIC_GetPriorityGrouping();
    
    NVIC_SetPriority(IRQn, NVIC_EncodePriority(prioritygroup, PreemptPriority, SubPriority));
  }
  
  /**
   * @brief Enables a device specific interrupt in the NVIC.
   * @param IRQn - External interrupt number
   */
  void HAL_NVIC_EnableIRQ(IRQn_Type IRQn)
  {
    /* Check the parameters */
    assert_param(IS_NVIC_DEVICE_IRQ(IRQn));
    
    /* Enable interrupt */
    NVIC_EnableIRQ(IRQn);
  }
  
  /**
   * @brief Disables a device specific interrupt in the NVIC.
   * @param IRQn - External interrupt number
   */
  void HAL_NVIC_DisableIRQ(IRQn_Type IRQn)
  {
    /* Check the parameters */
    assert_param(IS_NVIC_DEVICE_IRQ(IRQn));
    
    /* Disable interrupt */
    NVIC_DisableIRQ(IRQn);
  }
  
  /**
   * @brief Generate a Software interrupt on selected interrupt line.
   * @param IRQn - External interrupt number
   */
  void HAL_NVIC_SetPendingIRQ(IRQn_Type IRQn)
  {
    /* Check the parameters */
    assert_param(IS_NVIC_DEVICE_IRQ(IRQn));
    
    /* Set interrupt pending */
    NVIC_SetPendingIRQ(IRQn);
  }
  
  /**
   * @brief Clears the pending bit of a device specific interrupt in the NVIC.
   * @param IRQn - External interrupt number
   */
  void HAL_NVIC_ClearPendingIRQ(IRQn_Type IRQn)
  {
    /* Check the parameters */
    assert_param(IS_NVIC_DEVICE_IRQ(IRQn));
    
    /* Clear interrupt pending */
    NVIC_ClearPendingIRQ(IRQn);
  }
  
  /**
   * @brief Gets the pending status of a device specific interrupt in the NVIC.
   * @param IRQn - Device specific interrupt number
   * @return 0 - Interrupt status is not pending
   *         1 - Interrupt status is pending
   */
  uint32_t HAL_NVIC_GetPendingIRQ(IRQn_Type IRQn)
  {
    /* Check the parameters */
    assert_param(IS_NVIC_DEVICE_IRQ(IRQn));
    
    /* Return 1 if pending else 0 */
    return NVIC_GetPendingIRQ(IRQn);
  }
  
  /**
   * @brief Configures the SysTick clock source.
   * @param CLKSource - Specifies the SysTick clock source
   *         This parameter can be one of the following values:
   *           @arg SYSTICK_CLKSOURCE_HCLK_DIV8: AHB clock divided by 8 selected as SysTick clock source
   *           @arg SYSTICK_CLKSOURCE_HCLK: AHB clock selected as SysTick clock source
   */
  void HAL_SYSTICK_CLKSourceConfig(uint32_t CLKSource)
  {
    /* Check the parameters */
    assert_param(IS_SYSTICK_CLK_SOURCE(CLKSource));
    
    if (CLKSource == SYSTICK_CLKSOURCE_HCLK)
    {
      SysTick->CTRL |= SYSTICK_CLKSOURCE_HCLK;
    }
    else
    {
      SysTick->CTRL &= ~SYSTICK_CLKSOURCE_HCLK;
    }
  }
  
  /**
   * @brief This function handles the SysTick Handler, it only increments the tick counter.
   */
  void HAL_SYSTICK_IRQHandler(void)
  {
    HAL_IncTick();
  }
  
  /**
   * @brief System reset
   * @note This function performs a system reset by calling NVIC_SystemReset()
   */
  void HAL_NVIC_SystemReset(void)
  {
    /* System Reset */
    NVIC_SystemReset();
  }
  
  /**
   * @brief  Configures the SysTick clock for time interval in milliseconds
   * @param  TicksNumb: Specifies the ticks for time interval 
   * @retval HAL status
   */
  uint32_t HAL_SYSTICK_Config(uint32_t TicksNumb)
  {
    return SysTick_Config(TicksNumb);
  }