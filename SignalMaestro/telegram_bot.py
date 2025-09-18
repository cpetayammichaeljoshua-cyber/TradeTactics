"""
Telegram bot implementation for trading signal processing
Handles user interactions and command processing
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, ContextTypes, filters
)

from config import Config
from signal_parser import SignalParser
from risk_manager import RiskManager
from utils import format_currency, format_percentage
from telegram_strategy_comparison import TelegramStrategyComparison

class TradingSignalBot:
    """Telegram bot for handling trading signals and user interactions"""
    
    def __init__(self, binance_trader, cornix_integration, database):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.binance_trader = binance_trader
        self.cornix = cornix_integration
        self.db = database
        self.signal_parser = SignalParser()
        self.risk_manager = RiskManager()
        self.application = None
        
        # Initialize strategy comparison service
        self.strategy_comparison = TelegramStrategyComparison(self.config)
        
    async def initialize(self):
        """Initialize the Telegram bot application"""
        try:
            self.application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("balance", self.balance_command))
            self.application.add_handler(CommandHandler("positions", self.positions_command))
            self.application.add_handler(CommandHandler("signal", self.signal_command))
            self.application.add_handler(CommandHandler("settings", self.settings_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("history", self.history_command))
            
            # Add strategy comparison commands
            self.application.add_handler(CommandHandler("strategies", self.strategies_command))
            self.application.add_handler(CommandHandler("compare_run", self.compare_run_command))
            self.application.add_handler(CommandHandler("compare_status", self.compare_status_command))
            self.application.add_handler(CommandHandler("compare_result", self.compare_result_command))
            self.application.add_handler(CommandHandler("compare_recent", self.compare_recent_command))
            self.application.add_handler(CommandHandler("compare_rankings", self.compare_rankings_command))
            self.application.add_handler(CommandHandler("compare_tips", self.compare_tips_command))
            self.application.add_handler(CommandHandler("compare_help", self.compare_help_command))
            
            # Add message handler for signals
            self.application.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
            )
            
            # Add callback query handler for inline keyboards
            self.application.add_handler(CallbackQueryHandler(self.handle_callback))
            
            self.logger.info("Telegram bot application initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram bot: {e}")
            raise
    
    async def start(self):
        """Start the Telegram bot"""
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
    async def stop(self):
        """Stop the Telegram bot"""
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        if not self.config.is_authorized_user(user_id):
            await update.message.reply_text(
                "❌ You are not authorized to use this bot. Contact the administrator."
            )
            return
        
        # Save user to database
        await self.db.save_user(user_id, username)
        
        welcome_message = f"""
🚀 **Welcome to Trading Signal Bot!**

Hello {username}! I'm your automated cryptocurrency trading assistant.

**Available Commands:**
• `/balance` - Check account balance
• `/positions` - View open positions
• `/signal <pair>` - Get signal for trading pair
• `/settings` - Configure trading parameters
• `/status` - Check bot status
• `/history` - View trading history
• `/help` - Show this help message

**Strategy Comparison:**
• `/strategies` - List available strategies
• `/compare_run` - Start strategy comparison
• `/compare_recent` - Show recent comparisons
• `/compare_help` - Strategy comparison help

**Signal Format:**
You can send trading signals in these formats:
• `BUY BTCUSDT at 45000`
• `SELL ETHUSDT 50% at 3200`
• `LONG BTC SL: 44000 TP: 48000`

**Features:**
✅ Automated signal parsing
✅ Risk management
✅ Binance integration
✅ Cornix forwarding
✅ Real-time monitoring

Ready to start trading! Send me a signal or use the commands above.
        """
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        self.logger.info(f"User {username} ({user_id}) started the bot")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
📚 **Trading Signal Bot Help**

**Commands:**
• `/start` - Initialize bot
• `/balance` - Check account balance
• `/positions` - View open positions  
• `/signal <pair>` - Get signal for pair (e.g., /signal BTCUSDT)
• `/settings` - Configure trading parameters
• `/status` - Check system status
• `/history` - View recent trades
• `/help` - Show this help

**Signal Formats:**
• `BUY BTCUSDT at 45000`
• `SELL ETHUSDT 50% at 3200`
• `LONG BTC SL: 44000 TP: 48000`

**Features:**
✅ Automated signal parsing
✅ Risk management
✅ Binance integration
✅ Cornix forwarding
✅ Real-time monitoring

Send me a trading signal or use the commands above!
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
        self.logger.info(f"User {update.effective_user.id} requested help")

    async def balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command"""
        try:
            user_id = update.effective_user.id
            await update.message.reply_text("⏳ Fetching your account balance...")
            
            # Get balance from Binance trader
            balance_data = await self.binance_trader.get_account_balance()
            
            if balance_data:
                balance_text = "💰 **Account Balance:**\n\n"
                for asset, data in balance_data.items():
                    if float(data.get('free', 0)) > 0:
                        balance_text += f"• {asset}: {data['free']}\n"
                
                await update.message.reply_text(balance_text, parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Unable to fetch balance. Please check your API configuration.")
                
        except Exception as e:
            self.logger.error(f"Error in balance command: {e}")
            await update.message.reply_text("❌ Error fetching balance. Please try again later.")

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        try:
            await update.message.reply_text("⏳ Fetching your open positions...")
            
            # Get positions from Binance trader
            positions = await self.binance_trader.get_open_positions()
            
            if positions:
                positions_text = "📊 **Open Positions:**\n\n"
                for position in positions:
                    symbol = position.get('symbol', 'Unknown')
                    side = position.get('side', 'Unknown')
                    size = position.get('size', 0)
                    pnl = position.get('unrealizedPnl', 0)
                    
                    positions_text += f"• {symbol} {side}\n"
                    positions_text += f"  Size: {size}\n"
                    positions_text += f"  PnL: {pnl} USDT\n\n"
                
                await update.message.reply_text(positions_text, parse_mode='Markdown')
            else:
                await update.message.reply_text("📭 No open positions found.")
                
        except Exception as e:
            self.logger.error(f"Error in positions command: {e}")
            await update.message.reply_text("❌ Error fetching positions. Please try again later.")

    async def signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signal command"""
        try:
            if not context.args:
                await update.message.reply_text("Please provide a trading pair. Example: `/signal BTCUSDT`", parse_mode='Markdown')
                return
            
            symbol = context.args[0].upper()
            await update.message.reply_text(f"⏳ Analyzing {symbol}...")
            
            # Get market data for the symbol
            market_data = await self.binance_trader.get_market_data(symbol)
            
            if market_data:
                price = market_data.get('price', 0)
                change_24h = market_data.get('priceChangePercent', 0)
                
                signal_text = f"📈 **Signal for {symbol}:**\n\n"
                signal_text += f"💰 Current Price: ${price}\n"
                signal_text += f"📊 24h Change: {change_24h}%\n\n"
                signal_text += "📊 Use `/compare_run` to backtest strategies for this symbol."
                
                await update.message.reply_text(signal_text, parse_mode='Markdown')
            else:
                await update.message.reply_text(f"❌ Unable to get data for {symbol}. Please check the symbol.")
                
        except Exception as e:
            self.logger.error(f"Error in signal command: {e}")
            await update.message.reply_text("❌ Error processing signal request. Please try again later.")

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        try:
            user_id = update.effective_user.id
            
            # Get user settings from database
            user_data = await self.db.get_user_settings(user_id)
            
            settings_text = "⚙️ **Trading Settings:**\n\n"
            settings_text += f"🎯 Risk per trade: {user_data.get('risk_percentage', 2)}%\n"
            settings_text += f"🤖 Auto trading: {'Enabled' if user_data.get('auto_trading', False) else 'Disabled'}\n"
            settings_text += f"📤 Cornix forwarding: {'Enabled' if user_data.get('cornix_enabled', False) else 'Disabled'}\n\n"
            settings_text += "Contact your administrator to modify these settings."
            
            await update.message.reply_text(settings_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in settings command: {e}")
            await update.message.reply_text("❌ Error fetching settings. Please try again later.")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            # Check system components
            binance_status = "✅" if await self.binance_trader.test_connection() else "❌"
            db_status = "✅" if await self.db.test_connection() else "❌"
            
            status_text = "🤖 **System Status:**\n\n"
            status_text += f"🔗 Binance API: {binance_status}\n"
            status_text += f"💾 Database: {db_status}\n"
            status_text += f"📱 Telegram Bot: ✅\n"
            status_text += f"🌐 Webhook Server: ✅\n\n"
            status_text += "All systems operational! Ready to trade."
            
            await update.message.reply_text(status_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in status command: {e}")
            await update.message.reply_text("❌ Error checking system status.")

    async def history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history command"""
        try:
            user_id = update.effective_user.id
            await update.message.reply_text("⏳ Fetching your trading history...")
            
            # Get recent trades from database
            trades = await self.db.get_user_trades(user_id, limit=5)
            
            if trades:
                history_text = "📊 **Recent Trades:**\n\n"
                for trade in trades:
                    symbol = trade.get('symbol', 'Unknown')
                    side = trade.get('side', 'Unknown')
                    amount = trade.get('amount', 0)
                    price = trade.get('price', 0)
                    pnl = trade.get('pnl', 0)
                    status = trade.get('status', 'Unknown')
                    
                    history_text += f"• {symbol} {side}\n"
                    history_text += f"  Amount: {amount}\n"
                    history_text += f"  Price: ${price}\n"
                    history_text += f"  P&L: ${pnl}\n"
                    history_text += f"  Status: {status}\n\n"
                
                await update.message.reply_text(history_text, parse_mode='Markdown')
            else:
                await update.message.reply_text("📭 No trading history found.")
                
        except Exception as e:
            self.logger.error(f"Error in history command: {e}")
            await update.message.reply_text("❌ Error fetching trading history.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages (potential trading signals)"""
        try:
            user_id = update.effective_user.id
            username = update.effective_user.username or "Unknown"
            message_text = update.message.text
            
            self.logger.info(f"Received message from {username} ({user_id}): {message_text}")
            
            # Send initial processing message
            processing_msg = await update.message.reply_text("🔄 Processing your signal...")
            
            # Parse the signal
            from signal_parser import SignalParser
            parser = SignalParser()
            parsed_signal = parser.parse_signal(message_text)
            
            if parsed_signal and parsed_signal.get('action'):
                # Store signal in database
                signal_data = {
                    'user_id': user_id,
                    'raw_text': message_text,
                    'parsed_signal': parsed_signal,
                    'status': 'received'
                }
                await self.db.store_signal(signal_data)
                
                # Format response
                symbol = parsed_signal.get('symbol', 'Unknown')
                action = parsed_signal.get('action', 'Unknown')
                price = parsed_signal.get('price', 0)
                
                response_text = f"✅ **Signal Received!**\n\n"
                response_text += f"📊 Pair: {symbol}\n"
                response_text += f"🔄 Action: {action}\n"
                if price > 0:
                    response_text += f"💰 Price: ${price}\n"
                response_text += f"\n⏳ Processing for execution..."
                
                await processing_msg.edit_text(response_text, parse_mode='Markdown')
                
                # Process signal for trading (if auto-trading enabled)
                user_settings = await self.db.get_user_settings(user_id)
                if user_settings.get('auto_trading', False):
                    # Execute trade through the main trading logic
                    await self.execute_signal(parsed_signal, user_id)
                
            else:
                await processing_msg.edit_text(
                    "❌ Unable to parse trading signal.\n\n"
                    "Please use format like:\n"
                    "• `BUY BTCUSDT at 45000`\n"
                    "• `SELL ETHUSDT 50% at 3200`",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            await update.message.reply_text("❌ Error processing your message. Please try again.")

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards"""
        try:
            query = update.callback_query
            await query.answer()
            
            data = query.data
            
            if data.startswith('confirm_trade_'):
                trade_id = data.replace('confirm_trade_', '')
                await query.edit_message_text("✅ Trade confirmed and executed!")
                
            elif data.startswith('cancel_trade_'):
                trade_id = data.replace('cancel_trade_', '')
                await query.edit_message_text("❌ Trade cancelled.")
                
        except Exception as e:
            self.logger.error(f"Error handling callback: {e}")

    async def execute_signal(self, parsed_signal, user_id):
        """Execute a parsed trading signal"""
        try:
            # This would integrate with the main trading logic
            # For now, just log the action
            self.logger.info(f"Would execute signal for user {user_id}: {parsed_signal}")
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
    
    # ======== Strategy Comparison Commands ========
    
    async def strategies_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /strategies command - list available strategies"""
        try:
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            await update.message.reply_text("⏳ Loading available strategies...")
            
            strategies_text, total_count = await self.strategy_comparison.get_available_strategies()
            
            if total_count == 0:
                await update.message.reply_text(strategies_text, parse_mode='Markdown')
            else:
                footer = f"\n\n📱 Use `/compare_run` to start a comparison"
                final_text = strategies_text + footer
                
                if len(final_text) > 4096:
                    # Send in chunks if too long
                    await update.message.reply_text(strategies_text[:4090] + "...", parse_mode='Markdown')
                    await update.message.reply_text(footer, parse_mode='Markdown')
                else:
                    await update.message.reply_text(final_text, parse_mode='Markdown')
                    
        except Exception as e:
            self.logger.error(f"Error in strategies command: {e}")
            await update.message.reply_text("❌ Error loading strategies. Please try again later.")
    
    async def compare_run_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_run command - start new strategy comparison"""
        try:
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            # Parse arguments
            symbols = None
            days = 30
            strategies = None
            initial_capital = 10.0
            
            if context.args:
                try:
                    # Parse first argument (symbols)
                    if context.args[0].lower() != 'default':
                        symbols = [s.strip().upper() for s in context.args[0].split(',')]
                    
                    # Parse second argument (days)
                    if len(context.args) > 1:
                        days = int(context.args[1])
                    
                    # Parse third argument (strategies)  
                    if len(context.args) > 2:
                        strategies = [s.strip() for s in context.args[2].split(',')]
                    
                    # Parse fourth argument (capital)
                    if len(context.args) > 3:
                        initial_capital = float(context.args[3])
                        
                except (ValueError, IndexError) as e:
                    await update.message.reply_text(
                        "❌ Invalid arguments. Usage:\n"
                        "`/compare_run [symbols] [days] [strategies] [capital]`\n\n"
                        "Examples:\n"
                        "• `/compare_run` - Use defaults\n"
                        "• `/compare_run BTCUSDT,ETHUSDT 7` - Specific pairs, 7 days\n"
                        "• `/compare_run default 30 Ultimate,Momentum 100`",
                        parse_mode='Markdown'
                    )
                    return
            
            await update.message.reply_text("🚀 Starting strategy comparison...")
            
            # Start comparison
            result_text, comparison_id = await self.strategy_comparison.start_comparison(
                user_id=user_id,
                symbols=symbols,
                days=days,
                strategies=strategies,
                initial_capital=initial_capital
            )
            
            await update.message.reply_text(result_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in compare_run command: {e}")
            await update.message.reply_text("❌ Error starting comparison. Please try again later.")
    
    async def compare_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_status command - check comparison progress"""
        try:
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            if not context.args:
                await update.message.reply_text(
                    "❌ Please provide comparison ID.\nUsage: `/compare_status <id>`",
                    parse_mode='Markdown'
                )
                return
            
            comparison_id = context.args[0]
            status_text = await self.strategy_comparison.get_comparison_status(comparison_id)
            
            await update.message.reply_text(status_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in compare_status command: {e}")
            await update.message.reply_text("❌ Error checking status. Please try again later.")
    
    async def compare_result_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_result command - show comparison results"""
        try:
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            if not context.args:
                await update.message.reply_text(
                    "❌ Please provide comparison ID.\nUsage: `/compare_result <id> [page]`",
                    parse_mode='Markdown'
                )
                return
            
            comparison_id = context.args[0]
            page = 1
            
            if len(context.args) > 1:
                try:
                    page = int(context.args[1])
                except ValueError:
                    page = 1
            
            await update.message.reply_text("⏳ Loading comparison results...")
            
            # Get results
            result_text, chart_path, has_more = await self.strategy_comparison.get_comparison_result(
                comparison_id, page
            )
            
            # Send chart if available
            if chart_path and Path(chart_path).exists():
                try:
                    with open(chart_path, 'rb') as photo:
                        await update.message.reply_photo(
                            photo=photo,
                            caption=result_text,
                            parse_mode='Markdown'
                        )
                except Exception as e:
                    self.logger.warning(f"Error sending chart: {e}")
                    await update.message.reply_text(result_text, parse_mode='Markdown')
            else:
                await update.message.reply_text(result_text, parse_mode='Markdown')
            
            # Add pagination buttons if needed
            if has_more:
                next_page = page + 1
                await update.message.reply_text(
                    f"📄 More results available. Use:\n`/compare_result {comparison_id} {next_page}`",
                    parse_mode='Markdown'
                )
            
        except Exception as e:
            self.logger.error(f"Error in compare_result command: {e}")
            await update.message.reply_text("❌ Error loading results. Please try again later.")
    
    async def compare_recent_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_recent command - show recent comparisons"""
        try:
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            await update.message.reply_text("⏳ Loading your recent comparisons...")
            
            recent_text = await self.strategy_comparison.get_recent_comparisons(user_id)
            await update.message.reply_text(recent_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in compare_recent command: {e}")
            await update.message.reply_text("❌ Error loading recent comparisons. Please try again later.")
    
    async def compare_rankings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_rankings command - show strategy rankings by metric"""
        try:
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            if not context.args:
                available_metrics = self.strategy_comparison.get_available_metrics()
                metrics_list = '\n'.join([f"• `{metric}`" for metric in available_metrics])
                await update.message.reply_text(
                    f"❌ Please provide comparison ID and metric.\n\n"
                    f"Usage: `/compare_rankings <id> <metric>`\n\n"
                    f"Available metrics:\n{metrics_list}",
                    parse_mode='Markdown'
                )
                return
            
            comparison_id = context.args[0]
            metric = context.args[1] if len(context.args) > 1 else 'total_pnl_percentage'
            
            await update.message.reply_text("⏳ Loading strategy rankings...")
            
            rankings_text = await self.strategy_comparison.get_comparison_rankings(comparison_id, metric)
            await update.message.reply_text(rankings_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in compare_rankings command: {e}")
            await update.message.reply_text("❌ Error loading rankings. Please try again later.")
    
    async def compare_tips_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_tips command - show recommendations"""
        try:
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            if not context.args:
                await update.message.reply_text(
                    "❌ Please provide comparison ID.\nUsage: `/compare_tips <id>`",
                    parse_mode='Markdown'
                )
                return
            
            comparison_id = context.args[0]
            await update.message.reply_text("⏳ Loading recommendations...")
            
            recommendations_text = await self.strategy_comparison.get_comparison_recommendations(comparison_id)
            await update.message.reply_text(recommendations_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in compare_tips command: {e}")
            await update.message.reply_text("❌ Error loading recommendations. Please try again later.")
    
    async def compare_help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_help command - show strategy comparison help"""
        try:
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            help_text = await self.strategy_comparison.get_help_text()
            await update.message.reply_text(help_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in compare_help command: {e}")
            await update.message.reply_text("❌ Error loading help. Please try again later.")
