#!/usr/bin/env python3
"""
Dynamic 3-Level Stop Loss System
Implements sophisticated stop loss management with 3 independent levels that adapt to market conditions
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
from pathlib import Path


class StopLossLevel(Enum):
    """Stop loss level enumeration"""
    SL1 = "sl1"  # Tight stop loss (1-2%)
    SL2 = "sl2"  # Medium stop loss (3-5%)
    SL3 = "sl3"  # Wide stop loss (5-10%)


class MarketSession(Enum):
    """Market session types"""
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "overlap"


class VolatilityLevel(Enum):
    """Market volatility levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class StopLossConfig:
    """Configuration for stop loss levels"""
    # Base percentages for each level
    sl1_base_percent: float = 1.5  # 1.5% base for SL1
    sl2_base_percent: float = 4.0  # 4.0% base for SL2
    sl3_base_percent: float = 7.5  # 7.5% base for SL3
    
    # Volatility multipliers
    volatility_multipliers: Dict[VolatilityLevel, float] = None
    
    # Session adjustments
    session_adjustments: Dict[MarketSession, float] = None
    
    # Position size distribution when triggered
    sl1_position_percent: float = 33.0  # Close 33% at SL1
    sl2_position_percent: float = 33.0  # Close 33% at SL2
    sl3_position_percent: float = 34.0  # Close 34% at SL3
    
    # Trailing configuration
    trailing_enabled: bool = True
    trailing_distance_percent: float = 1.0  # Trail 1% behind favorable price
    
    def __post_init__(self):
        if self.volatility_multipliers is None:
            self.volatility_multipliers = {
                VolatilityLevel.LOW: 0.7,
                VolatilityLevel.MEDIUM: 1.0,
                VolatilityLevel.HIGH: 1.4,
                VolatilityLevel.EXTREME: 2.0
            }
        
        if self.session_adjustments is None:
            self.session_adjustments = {
                MarketSession.ASIA: 0.8,      # Lower volatility
                MarketSession.LONDON: 1.2,    # Higher volatility
                MarketSession.NEW_YORK: 1.0,  # Standard
                MarketSession.OVERLAP: 1.3    # Highest volatility
            }


@dataclass
class DynamicStopLoss:
    """Individual stop loss level with dynamic adjustment capabilities"""
    level: StopLossLevel
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    original_sl_price: float
    current_sl_price: float
    triggered: bool = False
    trigger_time: Optional[datetime] = None
    trail_high_water_mark: Optional[float] = None  # For long positions
    trail_low_water_mark: Optional[float] = None   # For short positions
    position_percent: float = 33.0  # Percentage of position this SL protects
    last_update_time: datetime = None
    
    def __post_init__(self):
        if self.last_update_time is None:
            self.last_update_time = datetime.now()
        
        # Initialize trail water marks
        if self.direction.lower() == 'long':
            self.trail_high_water_mark = self.current_price
        else:
            self.trail_low_water_mark = self.current_price


@dataclass
class MarketConditions:
    """Current market conditions for dynamic adjustments"""
    volatility_level: VolatilityLevel
    market_session: MarketSession
    atr_value: float
    volume_ratio: float
    momentum_strength: float
    support_resistance_distance: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MarketAnalyzer:
    """Analyzes market conditions for dynamic stop loss adjustments"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_current_session(self, dt: datetime = None) -> MarketSession:
        """Determine current market session"""
        if dt is None:
            dt = datetime.utcnow()
        
        hour = dt.hour
        
        # London session: 08:00 - 16:00 UTC
        # NY session: 13:00 - 21:00 UTC
        # Asia session: 21:00 - 05:00 UTC
        
        if 8 <= hour < 13:
            return MarketSession.LONDON
        elif 13 <= hour < 16:
            return MarketSession.OVERLAP  # London-NY overlap
        elif 16 <= hour < 21:
            return MarketSession.NEW_YORK
        else:
            return MarketSession.ASIA
    
    def calculate_volatility_level(self, atr_value: float, avg_atr: float) -> VolatilityLevel:
        """Calculate volatility level based on ATR"""
        if avg_atr <= 0:
            return VolatilityLevel.MEDIUM
        
        atr_ratio = atr_value / avg_atr
        
        if atr_ratio < 0.7:
            return VolatilityLevel.LOW
        elif atr_ratio < 1.3:
            return VolatilityLevel.MEDIUM
        elif atr_ratio < 2.0:
            return VolatilityLevel.HIGH
        else:
            return VolatilityLevel.EXTREME
    
    def analyze_market_conditions(self, symbol: str, price_data: List[Dict], 
                                current_price: float) -> MarketConditions:
        """Analyze current market conditions"""
        try:
            # Calculate ATR
            atr_value = self._calculate_atr(price_data)
            
            # Calculate average ATR for comparison
            avg_atr = self._calculate_average_atr(price_data, period=20)
            
            # Determine volatility level
            volatility_level = self.calculate_volatility_level(atr_value, avg_atr)
            
            # Get current session
            market_session = self.get_current_session()
            
            # Calculate volume ratio
            volume_ratio = self._calculate_volume_ratio(price_data)
            
            # Calculate momentum strength
            momentum_strength = self._calculate_momentum_strength(price_data, current_price)
            
            # Calculate distance to support/resistance
            support_resistance_distance = self._calculate_sr_distance(price_data, current_price)
            
            return MarketConditions(
                volatility_level=volatility_level,
                market_session=market_session,
                atr_value=atr_value,
                volume_ratio=volume_ratio,
                momentum_strength=momentum_strength,
                support_resistance_distance=support_resistance_distance
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {e}")
            # Return default conditions
            return MarketConditions(
                volatility_level=VolatilityLevel.MEDIUM,
                market_session=self.get_current_session(),
                atr_value=0.01,
                volume_ratio=1.0,
                momentum_strength=0.5,
                support_resistance_distance=0.02
            )
    
    def _calculate_atr(self, price_data: List[Dict], period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(price_data) < period + 1:
            return 0.01  # Default value
        
        true_ranges = []
        
        for i in range(1, len(price_data)):
            current = price_data[i]
            previous = price_data[i-1]
            
            high_low = current['high'] - current['low']
            high_close = abs(current['high'] - previous['close'])
            low_close = abs(current['low'] - previous['close'])
            
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
        
        if len(true_ranges) >= period:
            return sum(true_ranges[-period:]) / period
        else:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.01
    
    def _calculate_average_atr(self, price_data: List[Dict], period: int = 20) -> float:
        """Calculate average ATR over a longer period"""
        atr_values = []
        
        for i in range(14, len(price_data) - 14):
            segment = price_data[i-14:i+1]
            atr = self._calculate_atr(segment)
            atr_values.append(atr)
        
        return sum(atr_values) / len(atr_values) if atr_values else 0.01
    
    def _calculate_volume_ratio(self, price_data: List[Dict]) -> float:
        """Calculate current volume ratio to average"""
        if len(price_data) < 20:
            return 1.0
        
        recent_volumes = [candle.get('volume', 0) for candle in price_data[-20:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        
        current_volume = price_data[-1].get('volume', avg_volume)
        
        if avg_volume > 0:
            return current_volume / avg_volume
        return 1.0
    
    def _calculate_momentum_strength(self, price_data: List[Dict], current_price: float) -> float:
        """Calculate momentum strength (0-1 scale)"""
        if len(price_data) < 10:
            return 0.5
        
        # Simple momentum calculation based on price change
        price_10_ago = price_data[-10]['close']
        price_change = (current_price - price_10_ago) / price_10_ago
        
        # Normalize to 0-1 scale
        momentum = min(abs(price_change) * 50, 1.0)  # Scale by 50 to get reasonable values
        return momentum
    
    def _calculate_sr_distance(self, price_data: List[Dict], current_price: float) -> float:
        """Calculate distance to nearest support/resistance level"""
        if len(price_data) < 20:
            return 0.02
        
        # Simple S/R calculation using recent highs and lows
        recent_highs = [candle['high'] for candle in price_data[-20:]]
        recent_lows = [candle['low'] for candle in price_data[-20:]]
        
        resistance = max(recent_highs)
        support = min(recent_lows)
        
        # Distance to nearest level
        distance_to_resistance = abs(resistance - current_price) / current_price
        distance_to_support = abs(current_price - support) / current_price
        
        return min(distance_to_resistance, distance_to_support)


class TradeStopLossManager:
    """Manages all 3 stop loss levels for a single trade"""
    
    def __init__(self, 
                 symbol: str,
                 direction: str,
                 entry_price: float,
                 position_size: float,
                 config: StopLossConfig = None):
        self.symbol = symbol
        self.direction = direction.lower()
        self.entry_price = entry_price
        self.position_size = position_size
        self.config = config or StopLossConfig()
        
        self.logger = logging.getLogger(__name__)
        self.market_analyzer = MarketAnalyzer()
        
        # Initialize stop losses
        self.stop_losses: Dict[StopLossLevel, DynamicStopLoss] = {}
        self._initialize_stop_losses()
        
        # Trade state
        self.active = True
        self.remaining_position_size = position_size
        self.total_closed_amount = 0.0
        self.creation_time = datetime.now()
        self.last_update_time = datetime.now()
        
        # Performance tracking
        self.trigger_history: List[Dict[str, Any]] = []
        
    def _initialize_stop_losses(self):
        """Initialize all three stop loss levels"""
        # Calculate initial stop loss prices
        sl1_price = self._calculate_initial_sl_price(self.config.sl1_base_percent)
        sl2_price = self._calculate_initial_sl_price(self.config.sl2_base_percent)
        sl3_price = self._calculate_initial_sl_price(self.config.sl3_base_percent)
        
        # Create stop loss objects
        self.stop_losses[StopLossLevel.SL1] = DynamicStopLoss(
            level=StopLossLevel.SL1,
            symbol=self.symbol,
            direction=self.direction,
            entry_price=self.entry_price,
            current_price=self.entry_price,
            original_sl_price=sl1_price,
            current_sl_price=sl1_price,
            position_percent=self.config.sl1_position_percent
        )
        
        self.stop_losses[StopLossLevel.SL2] = DynamicStopLoss(
            level=StopLossLevel.SL2,
            symbol=self.symbol,
            direction=self.direction,
            entry_price=self.entry_price,
            current_price=self.entry_price,
            original_sl_price=sl2_price,
            current_sl_price=sl2_price,
            position_percent=self.config.sl2_position_percent
        )
        
        self.stop_losses[StopLossLevel.SL3] = DynamicStopLoss(
            level=StopLossLevel.SL3,
            symbol=self.symbol,
            direction=self.direction,
            entry_price=self.entry_price,
            current_price=self.entry_price,
            original_sl_price=sl3_price,
            current_sl_price=sl3_price,
            position_percent=self.config.sl3_position_percent
        )
        
        self.logger.info(f"Initialized 3-level stop loss for {self.symbol} {self.direction}")
        self.logger.info(f"SL1: {sl1_price:.6f}, SL2: {sl2_price:.6f}, SL3: {sl3_price:.6f}")
    
    def _calculate_initial_sl_price(self, base_percent: float) -> float:
        """Calculate initial stop loss price"""
        if self.direction == 'long':
            return self.entry_price * (1 - base_percent / 100)
        else:
            return self.entry_price * (1 + base_percent / 100)
    
    async def update_market_conditions(self, current_price: float, 
                                     price_data: List[Dict] = None) -> List[Dict[str, Any]]:
        """Update stop losses based on current market conditions"""
        triggered_levels = []
        
        try:
            # Update current price for all stop losses
            for sl in self.stop_losses.values():
                sl.current_price = current_price
                sl.last_update_time = datetime.now()
            
            # Analyze market conditions if price data available
            if price_data:
                market_conditions = self.market_analyzer.analyze_market_conditions(
                    self.symbol, price_data, current_price
                )
                await self._adjust_stop_losses_for_conditions(market_conditions)
            
            # Check for triggers
            triggered_levels = await self._check_stop_loss_triggers(current_price)
            
            # Update trailing stops
            if self.config.trailing_enabled:
                await self._update_trailing_stops(current_price)
            
            self.last_update_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating market conditions: {e}")
        
        return triggered_levels
    
    async def _adjust_stop_losses_for_conditions(self, conditions: MarketConditions):
        """Adjust stop loss levels based on market conditions"""
        try:
            # Calculate adjustment factors
            volatility_factor = self.config.volatility_multipliers.get(
                conditions.volatility_level, 1.0
            )
            session_factor = self.config.session_adjustments.get(
                conditions.market_session, 1.0
            )
            
            # Combined adjustment factor
            total_factor = volatility_factor * session_factor
            
            # Additional adjustments based on other conditions
            if conditions.volume_ratio > 2.0:  # High volume
                total_factor *= 1.1
            
            if conditions.momentum_strength > 0.8:  # Strong momentum
                total_factor *= 0.9  # Tighter stops
            
            # Apply adjustments to non-triggered stop losses
            for level, sl in self.stop_losses.items():
                if not sl.triggered:
                    # Calculate new stop loss price
                    base_percent = self._get_base_percent_for_level(level)
                    adjusted_percent = base_percent * total_factor
                    
                    new_sl_price = self._calculate_adjusted_sl_price(adjusted_percent)
                    
                    # Only move stop loss in favorable direction
                    if self._is_favorable_sl_move(sl.current_sl_price, new_sl_price):
                        old_price = sl.current_sl_price
                        sl.current_sl_price = new_sl_price
                        
                        self.logger.debug(
                            f"Adjusted {level.value} from {old_price:.6f} to {new_sl_price:.6f} "
                            f"(factor: {total_factor:.2f})"
                        )
            
        except Exception as e:
            self.logger.error(f"Error adjusting stop losses for conditions: {e}")
    
    def _get_base_percent_for_level(self, level: StopLossLevel) -> float:
        """Get base percentage for stop loss level"""
        base_percents = {
            StopLossLevel.SL1: self.config.sl1_base_percent,
            StopLossLevel.SL2: self.config.sl2_base_percent,
            StopLossLevel.SL3: self.config.sl3_base_percent
        }
        return base_percents.get(level, 2.0)
    
    def _calculate_adjusted_sl_price(self, adjusted_percent: float) -> float:
        """Calculate adjusted stop loss price"""
        if self.direction == 'long':
            return self.entry_price * (1 - adjusted_percent / 100)
        else:
            return self.entry_price * (1 + adjusted_percent / 100)
    
    def _is_favorable_sl_move(self, current_sl: float, new_sl: float) -> bool:
        """Check if stop loss move is favorable (reduces risk)"""
        if self.direction == 'long':
            return new_sl > current_sl  # Higher stop loss for long positions
        else:
            return new_sl < current_sl  # Lower stop loss for short positions
    
    async def _check_stop_loss_triggers(self, current_price: float) -> List[Dict[str, Any]]:
        """Check if any stop losses are triggered"""
        triggered_levels = []
        
        for level, sl in self.stop_losses.items():
            if sl.triggered:
                continue
            
            # Check if stop loss is triggered
            if self._is_stop_loss_triggered(sl, current_price):
                sl.triggered = True
                sl.trigger_time = datetime.now()
                
                # Calculate position amount to close
                close_amount = (sl.position_percent / 100) * self.remaining_position_size
                self.remaining_position_size -= close_amount
                self.total_closed_amount += close_amount
                
                trigger_info = {
                    'level': level.value,
                    'trigger_price': current_price,
                    'sl_price': sl.current_sl_price,
                    'close_amount': close_amount,
                    'remaining_position': self.remaining_position_size,
                    'trigger_time': sl.trigger_time,
                    'position_percent': sl.position_percent
                }
                
                triggered_levels.append(trigger_info)
                self.trigger_history.append(trigger_info)
                
                self.logger.warning(
                    f"🛑 {level.value.upper()} TRIGGERED for {self.symbol}: "
                    f"Price {current_price:.6f} hit SL {sl.current_sl_price:.6f}, "
                    f"Closing {close_amount:.6f} ({sl.position_percent}%)"
                )
                
                # Check if all stop losses triggered (full position closed)
                if self.remaining_position_size <= 0.01:  # Small threshold for floating point
                    self.active = False
                    self.logger.info(f"All stop losses triggered for {self.symbol}, trade closed")
        
        return triggered_levels
    
    def _is_stop_loss_triggered(self, sl: DynamicStopLoss, current_price: float) -> bool:
        """Check if specific stop loss is triggered"""
        if self.direction == 'long':
            return current_price <= sl.current_sl_price
        else:
            return current_price >= sl.current_sl_price
    
    async def _update_trailing_stops(self, current_price: float):
        """Update trailing stop losses"""
        for sl in self.stop_losses.values():
            if sl.triggered:
                continue
            
            try:
                if self.direction == 'long':
                    # Update high water mark
                    if sl.trail_high_water_mark is None or current_price > sl.trail_high_water_mark:
                        sl.trail_high_water_mark = current_price
                    
                    # Calculate trailing stop price
                    trailing_sl = sl.trail_high_water_mark * (1 - self.config.trailing_distance_percent / 100)
                    
                    # Move stop loss up if trailing price is higher
                    if trailing_sl > sl.current_sl_price:
                        sl.current_sl_price = trailing_sl
                        self.logger.debug(f"Trailed {sl.level.value} up to {trailing_sl:.6f}")
                
                else:  # short position
                    # Update low water mark
                    if sl.trail_low_water_mark is None or current_price < sl.trail_low_water_mark:
                        sl.trail_low_water_mark = current_price
                    
                    # Calculate trailing stop price
                    trailing_sl = sl.trail_low_water_mark * (1 + self.config.trailing_distance_percent / 100)
                    
                    # Move stop loss down if trailing price is lower
                    if trailing_sl < sl.current_sl_price:
                        sl.current_sl_price = trailing_sl
                        self.logger.debug(f"Trailed {sl.level.value} down to {trailing_sl:.6f}")
            
            except Exception as e:
                self.logger.error(f"Error updating trailing stop for {sl.level.value}: {e}")
    
    def get_stop_loss_status(self) -> Dict[str, Any]:
        """Get current status of all stop losses"""
        status = {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'active': self.active,
            'remaining_position_size': self.remaining_position_size,
            'total_closed_amount': self.total_closed_amount,
            'creation_time': self.creation_time.isoformat(),
            'last_update_time': self.last_update_time.isoformat(),
            'stop_losses': {}
        }
        
        for level, sl in self.stop_losses.items():
            status['stop_losses'][level.value] = {
                'original_price': sl.original_sl_price,
                'current_price': sl.current_sl_price,
                'triggered': sl.triggered,
                'trigger_time': sl.trigger_time.isoformat() if sl.trigger_time else None,
                'position_percent': sl.position_percent,
                'trail_high_water_mark': sl.trail_high_water_mark,
                'trail_low_water_mark': sl.trail_low_water_mark
            }
        
        return status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this stop loss system"""
        triggered_count = sum(1 for sl in self.stop_losses.values() if sl.triggered)
        
        # Calculate time active
        time_active = (datetime.now() - self.creation_time).total_seconds() / 3600  # hours
        
        return {
            'triggered_levels': triggered_count,
            'remaining_position_percent': (self.remaining_position_size / self.position_size) * 100,
            'closed_position_percent': (self.total_closed_amount / self.position_size) * 100,
            'time_active_hours': time_active,
            'trigger_history': self.trigger_history,
            'is_complete': not self.active
        }


class DynamicStopLossDatabase:
    """Database for tracking stop loss performance and analytics"""
    
    def __init__(self, db_path: str = "stop_loss_analytics.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize stop loss analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Stop loss triggers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sl_triggers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                sl_level TEXT NOT NULL,
                trigger_price REAL NOT NULL,
                sl_price REAL NOT NULL,
                position_percent REAL NOT NULL,
                trigger_time TIMESTAMP NOT NULL,
                market_session TEXT,
                volatility_level TEXT,
                profit_loss REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sl_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                total_triggers INTEGER NOT NULL,
                position_saved_percent REAL NOT NULL,
                time_active_hours REAL NOT NULL,
                effectiveness_score REAL NOT NULL,
                date DATE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_trigger(self, trigger_data: Dict[str, Any]) -> bool:
        """Log stop loss trigger event"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sl_triggers (
                    symbol, direction, entry_price, sl_level, trigger_price, sl_price,
                    position_percent, trigger_time, market_session, volatility_level, profit_loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trigger_data.get('symbol'),
                trigger_data.get('direction'),
                trigger_data.get('entry_price'),
                trigger_data.get('level'),
                trigger_data.get('trigger_price'),
                trigger_data.get('sl_price'),
                trigger_data.get('position_percent'),
                trigger_data.get('trigger_time'),
                trigger_data.get('market_session'),
                trigger_data.get('volatility_level'),
                trigger_data.get('profit_loss', 0.0)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Failed to log stop loss trigger: {e}")
            return False


# Global stop loss managers registry
_active_stop_loss_managers: Dict[str, TradeStopLossManager] = {}


def create_stop_loss_manager(symbol: str, direction: str, entry_price: float, 
                           position_size: float, config: StopLossConfig = None) -> TradeStopLossManager:
    """Create and register a new stop loss manager"""
    manager_id = f"{symbol}_{direction}_{int(entry_price * 1000000)}"
    
    manager = TradeStopLossManager(symbol, direction, entry_price, position_size, config)
    _active_stop_loss_managers[manager_id] = manager
    
    return manager


def get_stop_loss_manager(manager_id: str) -> Optional[TradeStopLossManager]:
    """Get stop loss manager by ID"""
    return _active_stop_loss_managers.get(manager_id)


def remove_stop_loss_manager(manager_id: str) -> bool:
    """Remove stop loss manager from registry"""
    if manager_id in _active_stop_loss_managers:
        del _active_stop_loss_managers[manager_id]
        return True
    return False


def get_all_active_managers() -> Dict[str, TradeStopLossManager]:
    """Get all active stop loss managers"""
    return _active_stop_loss_managers.copy()


def cleanup_inactive_managers():
    """Remove inactive stop loss managers"""
    inactive_managers = [
        manager_id for manager_id, manager in _active_stop_loss_managers.items()
        if not manager.active
    ]
    
    for manager_id in inactive_managers:
        del _active_stop_loss_managers[manager_id]
    
    return len(inactive_managers)