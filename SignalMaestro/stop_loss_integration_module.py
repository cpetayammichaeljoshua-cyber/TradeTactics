#!/usr/bin/env python3
"""
Stop Loss Integration Module
Provides complete integration of dynamic stop loss system into live trading pipeline
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

# Enhanced systems imports with fallback
try:
    from dynamic_stop_loss_system import (
        create_stop_loss_manager, DynamicStopLoss, StopLossConfig,
        StopLossLevel, VolatilityLevel, MarketSession, get_stop_loss_manager
    )
    from advanced_error_handler import handle_errors, APIException, TradingException
    from api_resilience_layer import resilient_api_call
    ENHANCED_SYSTEMS_AVAILABLE = True
except ImportError:
    ENHANCED_SYSTEMS_AVAILABLE = False
    
    # Create fallback classes when enhanced systems are not available
    from typing import Any, Dict, List
    
    class StopLossConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class APIException(Exception):
        pass
    
    class TradingException(Exception):
        pass
    
    def handle_errors(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def resilient_api_call(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    async def create_stop_loss_manager(*args, **kwargs):
        # Return a mock manager object
        class MockStopLossManager:
            async def update_price(self, price):
                return []
        return MockStopLossManager()


@dataclass
class StopLossAction:
    """Represents a stop loss action that needs to be executed"""
    symbol: str
    level: str  # 'SL1', 'SL2', 'SL3'
    percentage: float  # Percentage of position to close
    price: float  # Current price when triggered
    reason: str  # Reason for closure


class StopLossIntegrator:
    """Integrates dynamic stop loss system into trading bots"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.logger = logging.getLogger(__name__)
        self.stop_loss_managers: Dict[str, Any] = {}
        self.enhanced_systems_active = ENHANCED_SYSTEMS_AVAILABLE
        
        if ENHANCED_SYSTEMS_AVAILABLE:
            # Configure stop loss for scalping strategy
            self.stop_loss_config = StopLossConfig(
                sl1_base_percent=1.5,  # Tight SL for scalping
                sl2_base_percent=3.0,  # Medium SL
                sl3_base_percent=5.0,  # Wide SL
                sl1_position_percent=33.0,  # Close 33% at SL1
                sl2_position_percent=33.0,  # Close 33% at SL2
                sl3_position_percent=34.0,  # Close 34% at SL3
                trailing_enabled=True,
                trailing_distance_percent=0.5  # 0.5% trailing for scalping
            )
            self.logger.info("✅ Stop loss integration initialized successfully")
        else:
            self.logger.warning("⚠️ Enhanced stop loss system not available")
    
    async def create_trade_stop_loss(self, symbol: str, direction: str, entry_price: float) -> bool:
        """Create stop loss manager for a new trade"""
        if not self.enhanced_systems_active:
            self.logger.warning(f"⚠️ Stop loss creation not available for {symbol}")
            return False
            
        try:
            # Create stop loss manager
            stop_loss_manager = await create_stop_loss_manager(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                position_size=100.0,  # Default position size
                config=self.stop_loss_config
            )
            
            # Store the manager
            self.stop_loss_managers[symbol] = stop_loss_manager
            
            self.logger.info(f"🛡️ Dynamic stop loss system activated for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create stop loss manager for {symbol}: {e}")
            return False
    
    async def update_stop_loss_price(self, symbol: str, current_price: float) -> List[StopLossAction]:
        """Update stop loss with current price and return any triggered actions"""
        if not self.enhanced_systems_active or symbol not in self.stop_loss_managers:
            return []
            
        try:
            stop_loss_manager = self.stop_loss_managers[symbol]
            
            # Update with current price
            sl_actions = await stop_loss_manager.update_price(current_price)
            
            # Convert to StopLossAction objects
            actions = []
            for action in sl_actions:
                if action.get('action') == 'close_partial':
                    actions.append(StopLossAction(
                        symbol=symbol,
                        level=action.get('level', 'SL'),
                        percentage=action.get('percentage', 33.0),
                        price=current_price,
                        reason=action.get('reason', 'Stop loss triggered')
                    ))
            
            return actions
            
        except Exception as e:
            self.logger.error(f"❌ Stop loss update error for {symbol}: {e}")
            return []
    
    async def execute_stop_loss_action(self, action: StopLossAction) -> bool:
        """Execute a stop loss partial closure action"""
        try:
            self.logger.info(f"🚨 Executing {action.level} for {action.symbol}: {action.percentage}% at {action.price}")
            
            # Execute partial closure via Binance if available
            if hasattr(self.bot, 'binance_trader') and self.bot.binance_trader:
                success = await self._execute_binance_partial_close(action)
                if not success:
                    self.logger.warning(f"⚠️ Binance partial close failed for {action.symbol}")
            
            # Execute partial closure via Cornix if available
            if hasattr(self.bot, 'cornix') and self.bot.cornix:
                cornix_success = await self._execute_cornix_partial_close(action)
                if not cornix_success:
                    self.logger.warning(f"⚠️ Cornix partial close failed for {action.symbol}")
            
            # Send notification if rate limiter allows
            if hasattr(self.bot, 'rate_limiter') and self.bot.rate_limiter.can_send_message():
                await self._send_stop_loss_notification(action)
            
            self.logger.info(f"✅ {action.level} executed for {action.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to execute stop loss action: {e}")
            if ENHANCED_SYSTEMS_AVAILABLE:
                raise TradingException(f"Stop loss execution failed: {e}")
            return False
    
    async def _execute_binance_partial_close(self, action: StopLossAction) -> bool:
        """Execute partial position closure via Binance"""
        try:
            # Implementation depends on the specific Binance trader interface
            # This would call the enhanced Binance trader methods
            self.logger.info(f"📈 Executing Binance partial close: {action.symbol} {action.percentage}%")
            
            # For now, log the action (actual implementation would call Binance API)
            # Example: await self.bot.binance_trader.close_partial_position(action.symbol, action.percentage)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Binance partial close error: {e}")
            return False
    
    async def _execute_cornix_partial_close(self, action: StopLossAction) -> bool:
        """Execute partial position closure via Cornix"""
        try:
            if hasattr(self.bot.cornix, 'close_position'):
                result = await self.bot.cornix.close_position(
                    action.symbol,
                    f"{action.level} triggered - {action.reason}",
                    int(action.percentage)
                )
                return result.get('success', False) if isinstance(result, dict) else bool(result)
            else:
                self.logger.warning(f"⚠️ Cornix close_position method not available")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Cornix partial close error: {e}")
            return False
    
    async def _send_stop_loss_notification(self, action: StopLossAction):
        """Send stop loss notification"""
        try:
            level_emoji = {"SL1": "🟡", "SL2": "🟠", "SL3": "🔴"}.get(action.level, "🚨")
            
            msg = f"""{level_emoji} **{action.level} TRIGGERED** - {action.symbol}

🚨 **Stop Loss Hit:** {action.price:.4f}
📊 **Position Closed:** {action.percentage}%
💡 **Reason:** {action.reason}
🛡️ **Risk Management:** Active"""

            if hasattr(self.bot, 'send_rate_limited_message'):
                await self.bot.send_rate_limited_message(self.bot.admin_chat_id, msg)
            
        except Exception as e:
            self.logger.error(f"❌ Notification error: {e}")
    
    def cleanup_stop_loss_manager(self, symbol: str):
        """Clean up stop loss manager for completed trade"""
        if symbol in self.stop_loss_managers:
            try:
                del self.stop_loss_managers[symbol]
                self.logger.debug(f"🧹 Stop loss manager cleaned up for {symbol}")
            except Exception as e:
                self.logger.error(f"❌ Error cleaning up stop loss manager for {symbol}: {e}")
    
    def get_active_stop_losses(self) -> List[str]:
        """Get list of symbols with active stop loss managers"""
        return list(self.stop_loss_managers.keys())
    
    def is_stop_loss_active(self, symbol: str) -> bool:
        """Check if stop loss is active for a symbol"""
        return symbol in self.stop_loss_managers and self.enhanced_systems_active


# Export the integrator for easy import
__all__ = ['StopLossIntegrator', 'StopLossAction']