
import aiohttp
import logging
from typing import Optional, List, Dict


"""
Binance trading integration using ccxt library
Handles trade execution, portfolio management, and market data
ENHANCED with comprehensive error handling and resilience
"""

import asyncio
import logging
import ccxt.async_support as ccxt
from typing import Dict, Any, List, Optional
from decimal import Decimal, ROUND_DOWN
import time

from config import Config
from technical_analysis import TechnicalAnalysis

# Enhanced error handling imports
try:
    from advanced_error_handler import (
        handle_errors, RetryConfigs, CircuitBreaker,
        APIException, NetworkException, TradingException,
        InsufficientFundsException, RateLimitException
    )
    from api_resilience_layer import resilient_api_call, get_global_resilience_manager
    ENHANCED_ERROR_HANDLING = True
except ImportError:
    ENHANCED_ERROR_HANDLING = False
    # Create fallback decorators
    def handle_errors(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def resilient_api_call(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

class BinanceTrader:
    """Binance trading interface using ccxt with enhanced error handling"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.exchange = None
        self.technical_analysis = TechnicalAnalysis()
        
        # Enhanced error handling components
        self.enhanced_error_handling = ENHANCED_ERROR_HANDLING
        self.circuit_breaker = None
        self.resilience_manager = None
        
        if ENHANCED_ERROR_HANDLING:
            # Initialize circuit breaker for Binance API
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                timeout_duration=300,  # 5 minutes
                recovery_threshold=3
            )
            
            # Get global resilience manager
            self.resilience_manager = get_global_resilience_manager()
            
            self.logger.info("✅ Enhanced error handling enabled for Binance trader")
        else:
            self.logger.warning("⚠️ Enhanced error handling not available - using basic error handling")
        
    async def initialize(self):
        """Initialize Binance exchange connection"""
        try:
            # Use testnet if API keys are empty or testnet is enabled
            use_testnet = (not self.config.BINANCE_API_KEY or 
                          not self.config.BINANCE_API_SECRET or 
                          self.config.BINANCE_TESTNET)
            
            self.exchange = ccxt.binance({
                'apiKey': self.config.BINANCE_API_KEY or 'dummy_key',
                'secret': self.config.BINANCE_API_SECRET or 'dummy_secret',
                'sandbox': use_testnet,
                'timeout': self.config.BINANCE_REQUEST_TIMEOUT * 1000,
                'rateLimit': 1200,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            })
            
            if use_testnet:
                self.logger.info("Using Binance testnet (sandbox mode)")
            
            # Test connection
            await self.exchange.load_markets()
            self.logger.info("Binance exchange initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance exchange: {e}")
            raise
    
    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()
    
    async def ping(self) -> bool:
        """Test exchange connectivity"""
        try:
            # Try to fetch server time first (doesn't require API auth)
            await self.exchange.fetch_time()
            self.logger.info("Binance connection successful")
            return True
        except Exception as e:
            self.logger.warning(f"Binance ping failed: {e}")
            try:
                # Fallback: try to fetch ticker for BTCUSDT (public endpoint)
                await self.exchange.fetch_ticker('BTC/USDT')
                self.logger.info("Binance public API accessible")
                return True
            except Exception as e2:
                self.logger.error(f"Binance completely inaccessible: {e2}")
                return False
    
    @handle_errors(retry_config=RetryConfigs.API_RETRY)
    @resilient_api_call
    async def get_account_balance(self) -> Dict[str, Dict[str, float]]:
        """Get account balance for all assets with enhanced error handling"""
        try:
            # Use resilient API call if available
            if self.resilience_manager:
                balance = await self.resilience_manager.make_resilient_api_call(
                    "binance", self.exchange.fetch_balance
                )
            else:
                balance = await self.exchange.fetch_balance()
            
            # Filter out zero balances and format response
            filtered_balance = {}
            for symbol, data in balance['total'].items():
                if data > 0:
                    filtered_balance[symbol] = {
                        'free': balance['free'].get(symbol, 0),
                        'used': balance['used'].get(symbol, 0),
                        'total': data
                    }
            
            self.logger.debug(f"💰 Retrieved balance for {len(filtered_balance)} assets")
            return filtered_balance
            
        except Exception as e:
            self.logger.error(f"❌ Error getting account balance: {e}")
            if ENHANCED_ERROR_HANDLING:
                if 'rate limit' in str(e).lower():
                    raise RateLimitException("Rate limit exceeded getting account balance")
                elif 'network' in str(e).lower() or 'timeout' in str(e).lower():
                    raise NetworkException(f"Network error getting account balance: {e}")
                else:
                    raise APIException(f"API error getting account balance: {e}")
            raise
    
    async def get_portfolio_value(self) -> float:
        """Calculate total portfolio value in USDT"""
        try:
            balance = await self.get_account_balance()
            total_value = 0.0
            
            for symbol, data in balance.items():
                if symbol == 'USDT':
                    total_value += data['total']
                else:
                    # Convert to USDT value
                    try:
                        ticker_symbol = f"{symbol}/USDT"
                        if ticker_symbol in self.exchange.markets:
                            ticker = await self.exchange.fetch_ticker(ticker_symbol)
                            total_value += data['total'] * ticker['last']
                    except Exception:
                        # Skip if can't get price
                        continue
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    @handle_errors(retry_config=RetryConfigs.API_RETRY)
    @resilient_api_call
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol with enhanced error handling"""
        try:
            # Use resilient API call if available
            if self.resilience_manager:
                ticker = await self.resilience_manager.make_resilient_api_call(
                    "binance", self.exchange.fetch_ticker, symbol
                )
            else:
                ticker = await self.exchange.fetch_ticker(symbol)
                
            price = ticker.get('last') or ticker.get('close') or ticker.get('price')
            
            if price is None:
                self.logger.warning(f"⚠️ No price data available for {symbol}")
                if ENHANCED_ERROR_HANDLING:
                    raise APIException(f"No price data returned for {symbol}")
                return 0.0
                
            price_float = float(price)
            self.logger.debug(f"📊 Price for {symbol}: {price_float}")
            return price_float
            
        except Exception as e:
            self.logger.error(f"❌ Error getting price for {symbol}: {e}")
            if ENHANCED_ERROR_HANDLING:
                if 'rate limit' in str(e).lower():
                    raise RateLimitException(f"Rate limit exceeded getting price for {symbol}")
                elif 'network' in str(e).lower() or 'timeout' in str(e).lower():
                    raise NetworkException(f"Network error getting price for {symbol}: {e}")
                else:
                    raise APIException(f"API error getting price for {symbol}: {e}")
            return 0.0
    
    @handle_errors(retry_config=RetryConfigs.API_RETRY)
    @resilient_api_call
    async def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """Get OHLCV market data with enhanced error handling"""
        try:
            # Use resilient API call if available
            if self.resilience_manager:
                ohlcv = await self.resilience_manager.make_resilient_api_call(
                    "binance", self.exchange.fetch_ohlcv, symbol, timeframe, limit=limit
                )
            else:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Validate the data
            if not ohlcv or len(ohlcv) == 0:
                self.logger.warning(f"⚠️ No OHLCV data returned for {symbol} {timeframe}")
                if ENHANCED_ERROR_HANDLING:
                    raise APIException(f"No OHLCV data available for {symbol} {timeframe}")
                return []
                
            # Check if data contains valid values
            valid_data = []
            for candle in ohlcv:
                if len(candle) >= 6 and all(x is not None for x in candle[:6]):
                    valid_data.append(candle)
            
            if len(valid_data) == 0:
                self.logger.warning(f"⚠️ No valid OHLCV data for {symbol} {timeframe}")
                if ENHANCED_ERROR_HANDLING:
                    raise APIException(f"Invalid OHLCV data format for {symbol} {timeframe}")
                return []
                
            self.logger.debug(f"📊 Retrieved {len(valid_data)} candles for {symbol} {timeframe}")
            return valid_data
            
        except Exception as e:
            self.logger.error(f"❌ Error getting market data for {symbol} {timeframe}: {e}")
            if ENHANCED_ERROR_HANDLING:
                if 'rate limit' in str(e).lower():
                    raise RateLimitException(f"Rate limit exceeded getting market data for {symbol}")
                elif 'network' in str(e).lower() or 'timeout' in str(e).lower():
                    raise NetworkException(f"Network error getting market data: {e}")
                else:
                    raise APIException(f"API error getting market data: {e}")
            return []
    
    async def get_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get technical analysis for symbol"""
        try:
            # Get market data
            ohlcv_1h = await self.get_market_data(symbol, '1h', 100)
            ohlcv_4h = await self.get_market_data(symbol, '4h', 100)
            ohlcv_1d = await self.get_market_data(symbol, '1d', 50)
            
            # Check if we have valid market data
            if not ohlcv_1h and not ohlcv_4h and not ohlcv_1d:
                self.logger.warning(f"No market data available for {symbol}")
                return {'symbol': symbol, 'error': 'No market data available'}
            
            # Get 24h price change with fallback
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                price_change_24h = ticker.get('percentage', 0)
                volume = ticker.get('baseVolume', 0)
            except Exception as ticker_error:
                self.logger.warning(f"Error getting ticker data for {symbol}: {ticker_error}")
                price_change_24h = 0
                volume = 0
            
            # Calculate technical indicators
            analysis = await self.technical_analysis.analyze(
                ohlcv_1h, ohlcv_4h, ohlcv_1d
            )
            
            # Only add these if analysis was successful
            if 'error' not in analysis:
                analysis['price_change_24h'] = price_change_24h
                analysis['volume'] = volume
                analysis['symbol'] = symbol
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error getting technical analysis for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get open futures positions (if using futures)"""
        try:
            # For spot trading, we'll return current holdings as "positions"
            balance = await self.get_account_balance()
            positions = []
            
            for symbol, data in balance.items():
                if symbol != 'USDT' and data['total'] > 0:
                    try:
                        ticker_symbol = f"{symbol}/USDT"
                        if ticker_symbol in self.exchange.markets:
                            ticker = await self.exchange.fetch_ticker(ticker_symbol)
                            current_price = ticker['last']
                            
                            # Estimate unrealized PnL (simplified)
                            # This would be more accurate with actual entry prices
                            estimated_value = data['total'] * current_price
                            
                            positions.append({
                                'symbol': ticker_symbol,
                                'side': 'LONG',  # Spot holdings are always long
                                'size': data['total'],
                                'entryPrice': current_price,  # Simplified
                                'markPrice': current_price,
                                'unrealizedPnl': 0,  # Would need trade history for accurate PnL
                                'value': estimated_value
                            })
                    except Exception:
                        continue
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return []
    
    async def execute_trade(self, signal: Dict[str, Any], user_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade based on signal and user settings"""
        try:
            symbol = signal['symbol']
            action = signal['action'].upper()
            
            # Calculate position size
            position_size = await self._calculate_position_size(signal, user_settings)
            
            if position_size <= 0:
                return {
                    'success': False,
                    'error': 'Invalid position size calculated'
                }
            
            # Determine order type and price
            order_type = 'market'  # Default to market orders
            price = None
            
            if 'price' in signal and user_settings.get('use_limit_orders', False):
                order_type = 'limit'
                price = signal['price']
            
            # Execute the trade
            if action in ['BUY', 'LONG']:
                order = await self._execute_buy_order(symbol, position_size, order_type, price)
            elif action in ['SELL', 'SHORT']:
                order = await self._execute_sell_order(symbol, position_size, order_type, price)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported action: {action}'
                }
            
            if order:
                # Set stop loss and take profit if specified
                await self._set_stop_loss_take_profit(order, signal, user_settings)
                
                return {
                    'success': True,
                    'order_id': order['id'],
                    'symbol': symbol,
                    'side': action,
                    'amount': order['amount'],
                    'price': order.get('price', order.get('average', 0)),
                    'fee': order.get('fee', {}),
                    'timestamp': order['timestamp']
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to execute order'
                }
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _calculate_position_size(self, signal: Dict[str, Any], user_settings: Dict[str, Any]) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account balance
            balance = await self.get_account_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            # Get risk percentage from user settings
            risk_percentage = user_settings.get('risk_percentage', self.config.DEFAULT_RISK_PERCENTAGE)
            
            # Calculate risk amount
            risk_amount = usdt_balance * (risk_percentage / 100)
            
            # Get current price
            symbol = signal['symbol']
            current_price = await self.get_current_price(symbol)
            
            # If specific quantity is provided in signal, use it (with limits)
            if 'quantity' in signal:
                quantity = float(signal['quantity'])
                max_quantity = risk_amount / current_price
                return min(quantity, max_quantity)
            
            # Calculate position size based on stop loss
            if 'stop_loss' in signal:
                stop_loss = float(signal['stop_loss'])
                entry_price = signal.get('price', current_price)
                
                # Calculate risk per unit
                if signal['action'].upper() in ['BUY', 'LONG']:
                    risk_per_unit = abs(entry_price - stop_loss)
                else:
                    risk_per_unit = abs(stop_loss - entry_price)
                
                if risk_per_unit > 0:
                    quantity = risk_amount / risk_per_unit
                    # Convert to base currency quantity
                    position_size = min(quantity, risk_amount / current_price)
                else:
                    position_size = risk_amount / current_price
            else:
                # No stop loss specified, use full risk amount
                position_size = risk_amount / current_price
            
            # Apply position size limits
            max_position = user_settings.get('max_position_size', self.config.MAX_POSITION_SIZE)
            min_position = user_settings.get('min_position_size', self.config.MIN_POSITION_SIZE)
            
            position_value = position_size * current_price
            
            if position_value > max_position:
                position_size = max_position / current_price
            elif position_value < min_position:
                position_size = min_position / current_price
            
            # Round to appropriate precision
            market = self.exchange.markets.get(symbol, {})
            precision = market.get('precision', {}).get('amount', 8)
            
            return float(Decimal(str(position_size)).quantize(
                Decimal('0.' + '0' * (precision - 1) + '1'),
                rounding=ROUND_DOWN
            ))
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _execute_buy_order(self, symbol: str, amount: float, order_type: str, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Execute a buy order"""
        try:
            if order_type == 'market':
                order = await self.exchange.create_market_buy_order(symbol, amount)
            else:
                order = await self.exchange.create_limit_buy_order(symbol, amount, price)
            
            self.logger.info(f"Buy order executed: {order['id']} for {amount} {symbol}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
            return None
    
    async def _execute_sell_order(self, symbol: str, amount: float, order_type: str, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Execute a sell order"""
        try:
            if order_type == 'market':
                order = await self.exchange.create_market_sell_order(symbol, amount)
            else:
                order = await self.exchange.create_limit_sell_order(symbol, amount, price)
            
            self.logger.info(f"Sell order executed: {order['id']} for {amount} {symbol}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
            return None
    
    async def _set_stop_loss_take_profit(self, order: Dict[str, Any], signal: Dict[str, Any], user_settings: Dict[str, Any]):
        """Set stop loss and take profit orders"""
        try:
            symbol = order['symbol']
            amount = order['amount']
            side = 'sell' if signal['action'].upper() in ['BUY', 'LONG'] else 'buy'
            
            # Set stop loss
            if 'stop_loss' in signal:
                stop_loss_price = float(signal['stop_loss'])
                try:
                    stop_order = await self.exchange.create_order(
                        symbol=symbol,
                        type='stop_market',
                        side=side,
                        amount=amount,
                        params={'stopPrice': stop_loss_price}
                    )
                    self.logger.info(f"Stop loss set at {stop_loss_price} for order {order['id']}")
                except Exception as e:
                    self.logger.warning(f"Failed to set stop loss: {e}")
            
            # Set take profit
            if 'take_profit' in signal:
                take_profit_price = float(signal['take_profit'])
                try:
                    tp_order = await self.exchange.create_limit_order(
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=take_profit_price
                    )
                    self.logger.info(f"Take profit set at {take_profit_price} for order {order['id']}")
                except Exception as e:
                    self.logger.warning(f"Failed to set take profit: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error setting stop loss/take profit: {e}")
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status"""
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            return order
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {}
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Order {order_id} cancelled successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for symbol"""
        try:
            fees = await self.exchange.fetch_trading_fee(symbol)
            return {
                'maker': fees.get('maker', 0.001),
                'taker': fees.get('taker', 0.001)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading fees: {e}")
            return {'maker': 0.001, 'taker': 0.001}
    
    async def get_market_summary(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market summary for multiple symbols"""
        try:
            summaries = {}
            
            for symbol in symbols:
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    summaries[symbol] = {
                        'price': ticker['last'],
                        'change_24h': ticker['percentage'],
                        'volume': ticker['baseVolume'],
                        'high_24h': ticker['high'],
                        'low_24h': ticker['low']
                    }
                except Exception:
                    continue
            
            return summaries
            
        except Exception as e:
            self.logger.error(f"Error getting market summary: {e}")
            return {}
#!/usr/bin/env python3
"""
Binance Trading Integration
"""

import asyncio
import logging
import aiohttp
import json
from typing import Dict, Any, Optional, List

class BinanceTrader:
    """Binance trading interface"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_url = "https://api.binance.com"
        self.api_key = None
        self.api_secret = None
        
    async def test_connection(self) -> bool:
        """Test Binance API connection"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/api/v3/ping") as response:
                    return response.status == 200
        except:
            return False
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/api/v3/ticker/price?symbol={symbol}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data['price'])
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
        return None
    
    async def get_multi_timeframe_data(self, symbol: str, timeframes: List[str], limit: int = 100) -> Dict[str, List]:
        """Get OHLCV data for multiple timeframes"""
        try:
            data = {}
            
            for tf in timeframes:
                # Convert timeframe format
                binance_tf = self._convert_timeframe(tf)
                
                async with aiohttp.ClientSession() as session:
                    url = f"{self.api_url}/api/v3/klines?symbol={symbol}&interval={binance_tf}&limit={limit}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            klines = await response.json()
                            # Convert to OHLCV format
                            ohlcv = [[
                                kline[0],  # timestamp
                                float(kline[1]),  # open
                                float(kline[2]),  # high
                                float(kline[3]),  # low
                                float(kline[4]),  # close
                                float(kline[5])   # volume
                            ] for kline in klines]
                            data[tf] = ohlcv
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting multi-timeframe data: {e}")
            return {}
    
    def _convert_timeframe(self, tf: str) -> str:
        """Convert timeframe to Binance format"""
        mapping = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        return mapping.get(tf, '5m')
    
    async def get_ohlcv_data(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> List[List]:
        """Get OHLCV data for a single timeframe"""
        try:
            binance_tf = self._convert_timeframe(timeframe)
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_url}/api/v3/klines?symbol={symbol}&interval={binance_tf}&limit={limit}"
                async with session.get(url) as response:
                    if response.status == 200:
                        klines = await response.json()
                        # Convert to OHLCV format
                        ohlcv = [[
                            kline[0],  # timestamp
                            float(kline[1]),  # open
                            float(kline[2]),  # high
                            float(kline[3]),  # low
                            float(kline[4]),  # close
                            float(kline[5])   # volume
                        ] for kline in klines]
                        return ohlcv
            return []
        except Exception as e:
            self.logger.error(f"Error getting OHLCV data for {symbol} {timeframe}: {e}")
            return []
    
    async def close(self):
        """Close any connections"""
        # Placeholder for closing connections if needed
        pass
