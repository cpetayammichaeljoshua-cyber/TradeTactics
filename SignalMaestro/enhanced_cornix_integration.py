
"""
Enhanced Cornix Integration with Advanced SL/TP Management
Handles dynamic stop loss updates and position management
"""

import aiohttp
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from config import Config

class EnhancedCornixIntegration:
    """Enhanced Cornix integration with advanced trade management"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.webhook_url = self.config.CORNIX_WEBHOOK_URL
        self.bot_uuid = self.config.CORNIX_BOT_UUID
        
    async def send_initial_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Send initial trading signal to Cornix with proper formatting"""
        try:
            # Format for Cornix webhook
            cornix_payload = {
                'uuid': self.bot_uuid,
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'enhanced_perfect_scalping_bot',
                'action': signal['action'].lower(),
                'symbol': signal['symbol'].replace('USDT', '/USDT'),
                'entry_price': float(signal['entry_price']),
                'stop_loss': float(signal['stop_loss']),
                'take_profit_1': float(signal['tp1']),
                'take_profit_2': float(signal['tp2']),
                'take_profit_3': float(signal['tp3']),
                'leverage': signal.get('leverage', 10),
                'exchange': 'binance_futures',
                'type': 'futures',
                'margin_type': 'cross',
                'position_size_percentage': 100,
                
                # Enhanced TP distribution
                'tp_distribution': [40, 35, 25],  # 40% at TP1, 35% at TP2, 25% at TP3
                
                # Advanced SL management
                'sl_management': {
                    'move_to_entry_on_tp1': True,
                    'move_to_tp1_on_tp2': True,
                    'close_all_on_tp3': True,
                    'auto_sl_updates': True
                },
                
                # Signal metadata
                'risk_reward': signal.get('risk_reward_ratio', 3.0),
                'signal_strength': signal.get('signal_strength', 85),
                'strategy': 'Enhanced Perfect Scalping',
                'timeframe': 'Multi-TF',
                'auto_execute': True
            }
            
            result = await self._send_webhook(cornix_payload)
            
            if result.get('status') == 'success':
                self.logger.info(f"✅ Initial signal sent to Cornix: {signal['symbol']}")
                return {'success': True, 'response': result}
            else:
                self.logger.warning(f"⚠️ Cornix signal failed: {result}")
                return {'success': False, 'error': result.get('error')}
                
        except Exception as e:
            self.logger.error(f"❌ Error sending signal to Cornix: {e}")
            return {'success': False, 'error': str(e)}
    
    async def update_stop_loss(self, symbol: str, new_sl: float, reason: str) -> Dict[str, Any]:
        """Send stop loss update to Cornix"""
        try:
            update_payload = {
                'uuid': self.bot_uuid,
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'update_stop_loss',
                'symbol': symbol.replace('USDT', '/USDT'),
                'new_stop_loss': new_sl,
                'reason': reason,
                'update_type': 'trailing_sl',
                'auto_execute': True,
                'priority': 'high'
            }
            
            result = await self._send_webhook(update_payload)
            
            if result.get('status') == 'success':
                self.logger.info(f"✅ SL update sent to Cornix: {symbol} -> {new_sl}")
                return {'success': True, 'response': result}
            else:
                self.logger.warning(f"⚠️ Cornix SL update failed: {result}")
                return {'success': False, 'error': result.get('error')}
                
        except Exception as e:
            self.logger.error(f"❌ Error updating SL in Cornix: {e}")
            return {'success': False, 'error': str(e)}
    
    async def close_position(self, symbol: str, reason: str, percentage: int = 100) -> Dict[str, Any]:
        """Send position closure to Cornix"""
        try:
            closure_payload = {
                'uuid': self.bot_uuid,
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'close_position',
                'symbol': symbol.replace('USDT', '/USDT'),
                'close_percentage': percentage,
                'reason': reason,
                'close_type': 'market_order',
                'auto_execute': True,
                'priority': 'high',
                'final_closure': percentage == 100
            }
            
            result = await self._send_webhook(closure_payload)
            
            if result.get('status') == 'success':
                self.logger.info(f"✅ Position closure sent to Cornix: {symbol} ({percentage}%)")
                return {'success': True, 'response': result}
            else:
                self.logger.warning(f"⚠️ Cornix closure failed: {result}")
                return {'success': False, 'error': result.get('error')}
                
        except Exception as e:
            self.logger.error(f"❌ Error closing position in Cornix: {e}")
            return {'success': False, 'error': str(e)}
    
    async def partial_take_profit(self, symbol: str, tp_level: int, percentage: int) -> Dict[str, Any]:
        """Send partial take profit to Cornix"""
        try:
            tp_payload = {
                'uuid': self.bot_uuid,
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'partial_take_profit',
                'symbol': symbol.replace('USDT', '/USDT'),
                'tp_level': tp_level,
                'close_percentage': percentage,
                'reason': f'TP{tp_level} hit - partial closure',
                'auto_execute': True,
                'update_remaining_sl': True
            }
            
            result = await self._send_webhook(tp_payload)
            
            if result.get('status') == 'success':
                self.logger.info(f"✅ Partial TP sent to Cornix: {symbol} TP{tp_level} ({percentage}%)")
                return {'success': True, 'response': result}
            else:
                self.logger.warning(f"⚠️ Cornix partial TP failed: {result}")
                return {'success': False, 'error': result.get('error')}
                
        except Exception as e:
            self.logger.error(f"❌ Error sending partial TP to Cornix: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _send_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send webhook request to Cornix with enhanced error handling"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'EnhancedPerfectScalpingBot/2.0'
            }
            
            # Add authentication if configured
            if self.config.WEBHOOK_SECRET:
                headers['Authorization'] = f"Bearer {self.config.WEBHOOK_SECRET}"
            
            timeout = aiohttp.ClientTimeout(total=15)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers
                ) as response:
                    
                    response_text = await response.text()
                    
                    if response.status == 200:
                        try:
                            response_data = json.loads(response_text)
                            return {
                                'status': 'success',
                                'status_code': response.status,
                                'response': response_data
                            }
                        except json.JSONDecodeError:
                            return {
                                'status': 'success',
                                'status_code': response.status,
                                'response': response_text
                            }
                    else:
                        return {
                            'status': 'error',
                            'status_code': response.status,
                            'error': response_text
                        }
                        
        except aiohttp.ClientError as e:
            return {
                'status': 'error',
                'error': f'Network error: {str(e)}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Unexpected error: {str(e)}'
            }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Cornix webhook connection"""
        try:
            test_payload = {
                'type': 'connection_test',
                'timestamp': datetime.utcnow().isoformat(),
                'uuid': self.bot_uuid,
                'source': 'enhanced_perfect_scalping_bot',
                'test_message': 'Enhanced bot connection test'
            }
            
            result = await self._send_webhook(test_payload)
            
            return {
                'success': result.get('status') == 'success',
                'response': result,
                'webhook_url': self.webhook_url,
                'bot_uuid': self.bot_uuid
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def format_tradingview_alert(self, signal: Dict[str, Any]) -> str:
        """Format signal as TradingView alert for Cornix"""
        try:
            parts = [
                f"uuid={self.bot_uuid}",
                f"action={signal['action'].lower()}",
                f"symbol={signal['symbol']}",
                f"price={signal['entry_price']}",
                f"sl={signal['stop_loss']}",
                f"tp1={signal['tp1']}",
                f"tp2={signal['tp2']}",
                f"tp3={signal['tp3']}"
            ]
            
            return "\n".join(parts)
            
        except Exception as e:
            self.logger.error(f"❌ Error formatting TradingView alert: {e}")
            return ""
