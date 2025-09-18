"""
Cornix integration for signal forwarding and automation
Handles webhook communication with Cornix platform
"""

import aiohttp
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from config import Config

class CornixIntegration:
    """Integration with Cornix trading automation platform"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.webhook_url = self.config.CORNIX_WEBHOOK_URL
        self.bot_uuid = self.config.CORNIX_BOT_UUID
        
    async def forward_signal(self, signal: Dict[str, Any], trade_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Forward trading signal to Cornix platform
        
        Args:
            signal: Parsed trading signal
            trade_result: Optional trade execution result
            
        Returns:
            Dict with forwarding result
        """
        try:
            # Format signal for Cornix
            cornix_payload = self._format_signal_for_cornix(signal, trade_result)
            
            if not cornix_payload:
                return {
                    'success': False,
                    'error': 'Failed to format signal for Cornix'
                }
            
            # Send to Cornix webhook
            result = await self._send_webhook(cornix_payload)
            
            self.logger.info(f"Signal forwarded to Cornix: {signal['symbol']} {signal['action']}")
            
            return {
                'success': True,
                'cornix_response': result,
                'payload': cornix_payload
            }
            
        except Exception as e:
            self.logger.error(f"Error forwarding signal to Cornix: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _format_signal_for_cornix(self, signal: Dict[str, Any], trade_result: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Format signal for Cornix webhook format"""
        try:
            # Base payload structure for Cornix
            payload: Dict[str, Any] = {
                'uuid': self.bot_uuid,
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'perfect_scalping_bot'
            }
            
            # Map signal action to Cornix format
            action_mapping = {
                'BUY': 'buy',
                'SELL': 'sell', 
                'LONG': 'buy',
                'SHORT': 'sell'
            }
            
            action = signal.get('direction', signal.get('action', '')).upper()
            if action not in action_mapping:
                self.logger.warning(f"Unsupported action for Cornix: {action}")
                return None
            
            payload['action'] = action_mapping[action]
            payload['symbol'] = signal.get('symbol', '')
            
            # Add entry price
            if 'entry_price' in signal:
                payload['entry_price'] = float(signal['entry_price'])
            elif 'price' in signal:
                payload['entry_price'] = float(signal['price'])
            
            # Add stop loss and take profits
            if 'stop_loss' in signal:
                payload['stop_loss'] = float(signal['stop_loss'])
            
            # Multiple take profits
            if 'tp1' in signal:
                payload['take_profit_1'] = float(signal['tp1'])
            if 'tp2' in signal:
                payload['take_profit_2'] = float(signal['tp2'])
            if 'tp3' in signal:
                payload['take_profit_3'] = float(signal['tp3'])
            
            # Add leverage
            if 'optimal_leverage' in signal:
                payload['leverage'] = int(signal['optimal_leverage'])
            elif 'leverage' in signal:
                payload['leverage'] = int(signal['leverage'])
            
            # Trading parameters
            payload['exchange'] = 'binance'
            payload['type'] = 'futures'
            payload['margin_type'] = 'cross'
            
            # Signal metadata
            payload['signal_strength'] = signal.get('signal_strength', 0)
            payload['risk_reward_ratio'] = signal.get('risk_reward_ratio', 3.0)
            payload['timeframe'] = signal.get('timeframe', 'Multi-TF')
            payload['strategy'] = signal.get('strategy', 'Perfect Scalping')
            
            # Validate price relationships
            entry = payload.get('entry_price', 0)
            stop_loss = payload.get('stop_loss', 0)
            tp1 = payload.get('take_profit_1', 0)
            
            if entry == 0 or stop_loss == 0 or tp1 == 0:
                self.logger.warning("Missing essential price data for Cornix")
                return None
            
            # Final validation
            if action == 'BUY' and not (float(stop_loss) < float(entry) < float(tp1)):
                self.logger.warning("Invalid BUY price structure for Cornix")
                return None
            elif action == 'SELL' and not (float(tp1) < float(entry) < float(stop_loss)):
                self.logger.warning("Invalid SELL price structure for Cornix")
                return None
            
            return payload
            
        except Exception as e:
            self.logger.error(f"Error formatting signal for Cornix: {e}")
            return None
    
    async def _send_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send webhook request to Cornix"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'TradingSignalBot/1.0'
            }
            
            # Add authentication if webhook secret is configured
            if self.config.WEBHOOK_SECRET:
                headers['Authorization'] = f"Bearer {self.config.WEBHOOK_SECRET}"
            
            timeout = aiohttp.ClientTimeout(total=30)
            
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
                        self.logger.warning(f"Cornix webhook returned status {response.status}: {response_text}")
                        return {
                            'status': 'error',
                            'status_code': response.status,
                            'error': response_text
                        }
                        
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error sending webhook to Cornix: {e}")
            return {
                'status': 'error',
                'error': f'Network error: {str(e)}'
            }
        except Exception as e:
            self.logger.error(f"Unexpected error sending webhook to Cornix: {e}")
            return {
                'status': 'error',
                'error': f'Unexpected error: {str(e)}'
            }
    
    async def create_cornix_bot(self, bot_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new bot configuration in Cornix (if API supports it)"""
        try:
            # This would be used if Cornix provides an API for bot creation
            # Currently, bots are typically created through the web interface
            
            payload = {
                'name': bot_config.get('name', 'Telegram Signal Bot'),
                'exchange': bot_config.get('exchange', 'binance'),
                'trading_mode': bot_config.get('trading_mode', 'spot'),
                'risk_management': {
                    'max_position_size': bot_config.get('max_position_size', 1000),
                    'stop_loss_percentage': bot_config.get('stop_loss_percentage', 3),
                    'take_profit_percentage': bot_config.get('take_profit_percentage', 6)
                }
            }
            
            # This is a placeholder - actual implementation would depend on Cornix API
            self.logger.info("Bot creation would be implemented here if Cornix API supports it")
            
            return {
                'success': True,
                'bot_uuid': 'placeholder-uuid',
                'message': 'Bot creation placeholder - configure manually in Cornix dashboard'
            }
            
        except Exception as e:
            self.logger.error(f"Error creating Cornix bot: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_bot_status(self) -> Dict[str, Any]:
        """Get bot status from Cornix (if API supports it)"""
        try:
            # Placeholder for bot status check
            # This would query Cornix API for bot status, active positions, etc.
            
            return {
                'status': 'active',
                'bot_uuid': self.bot_uuid,
                'last_signal': 'N/A',
                'active_positions': 0,
                'total_trades': 0,
                'success_rate': 0,
                'message': 'Status check placeholder - implement with actual Cornix API'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting bot status: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def validate_webhook_config(self) -> Dict[str, Any]:
        """Validate webhook configuration"""
        try:
            validation_result: Dict[str, Any] = {
                'webhook_url_configured': bool(self.webhook_url),
                'bot_uuid_configured': bool(self.bot_uuid),
                'webhook_secret_configured': bool(self.config.WEBHOOK_SECRET)
            }
            
            # Test webhook connectivity (optional)
            if validation_result['webhook_url_configured']:
                try:
                    # Send a test ping to the webhook
                    test_payload = {
                        'type': 'test',
                        'timestamp': datetime.utcnow().isoformat(),
                        'message': 'Configuration test'
                    }
                    
                    result = await self._send_webhook(test_payload)
                    validation_result['webhook_reachable'] = result.get('status') == 'success'
                    validation_result['test_response'] = result
                    
                except Exception as e:
                    validation_result['webhook_reachable'] = False
                    validation_result['test_error'] = str(e)
            else:
                validation_result['webhook_reachable'] = False
            
            # Overall configuration status
            validation_result['configured'] = (
                validation_result['webhook_url_configured'] and 
                validation_result['bot_uuid_configured']
            )
            
            validation_result['ready'] = (
                validation_result['configured'] and 
                validation_result.get('webhook_reachable', False)
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating webhook config: {e}")
            return {
                'configured': False,
                'ready': False,
                'error': str(e)
            }
    
    def format_cornix_alert_message(self, signal: Dict[str, Any]) -> str:
        """
        Format signal as TradingView alert message for Cornix
        This can be used when setting up TradingView alerts
        """
        try:
            # Standard Cornix TradingView alert format
            message_parts = []
            
            # Add bot UUID
            if self.bot_uuid:
                message_parts.append(f"uuid={self.bot_uuid}")
            
            # Add action
            action = signal.get('action', '').lower()
            if action in ['buy', 'long']:
                message_parts.append("action=buy")
            elif action in ['sell', 'short']:
                message_parts.append("action=sell")
            
            # Add symbol
            symbol = signal.get('symbol', '')
            if symbol:
                message_parts.append(f"symbol={symbol}")
            
            # Add price
            if 'price' in signal:
                message_parts.append(f"price={signal['price']}")
            
            # Add quantity
            if 'quantity' in signal:
                message_parts.append(f"quantity={signal['quantity']}")
            
            # Add stop loss
            if 'stop_loss' in signal:
                message_parts.append(f"sl={signal['stop_loss']}")
            
            # Add take profit  
            if 'take_profit' in signal:
                message_parts.append(f"tp={signal['take_profit']}")
            
            # Join all parts
            alert_message = "\n".join(message_parts)
            
            return alert_message
            
        except Exception as e:
            self.logger.error(f"Error formatting Cornix alert message: {e}")
            return ""
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to Cornix webhook"""
        try:
            test_payload = {
                'type': 'connection_test',
                'timestamp': datetime.utcnow().isoformat(),
                'uuid': self.bot_uuid,
                'message': 'Testing connection from Telegram Trading Bot'
            }
            
            result = await self._send_webhook(test_payload)
            
            return {
                'success': result.get('status') == 'success',
                'response': result,
                'webhook_url': self.webhook_url,
                'bot_uuid': self.bot_uuid
            }
            
        except Exception as e:
            self.logger.error(f"Error testing Cornix connection: {e}")
            return {
                'success': False,
                'error': str(e)
            }
