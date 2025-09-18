
#!/usr/bin/env python3
"""
Adaptive ML Trading Bot Startup Script
Launches the most advanced trading bot with machine learning and adaptive algorithms
"""

import asyncio
import logging
import os
import sys
import signal
import time
from pathlib import Path

# Ensure the SignalMaestro directory is in the Python path
sys.path.append(str(Path(__file__).parent / "SignalMaestro"))

def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('adaptive_ml_bot.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

async def main():
    """Main function to run the adaptive ML trading bot"""
    logger = setup_logging()
    
    print("🚀 Adaptive ML Trading Bot Manager Starting")
    print("⚙️ Auto-restart enabled (max 10 restarts)")
    print("🧠 Machine Learning features active")
    print("🤖 Adaptive algorithms enabled")
    print("⏰ 30-minute cooldown system enabled")
    print("📊 Multi-timeframe analysis active")
    print("🎯 Dynamic SL/TP optimization")
    
    restart_count = 0
    max_restarts = 10
    
    while restart_count < max_restarts:
        try:
            logger.info("Starting Adaptive ML Trading Bot...")
            
            # Import and run the bot
            from ml_enhanced_trading_bot import MLEnhancedTradingBot, main as bot_main
            
            # Set environment variable to indicate adaptive mode
            os.environ['ADAPTIVE_ML_MODE'] = 'true'
            
            success = await bot_main()
            
            if success:
                logger.info("Bot completed successfully")
                break
            else:
                logger.warning("Bot returned error status")
                
        except KeyboardInterrupt:
            logger.info("🛑 Manual shutdown requested")
            break
        except Exception as e:
            restart_count += 1
            logger.error(f"Bot crashed: {e}")
            
            if restart_count < max_restarts:
                wait_time = min(60 * restart_count, 300)  # Max 5 minutes
                logger.info(f"Restarting in {wait_time} seconds... (Attempt {restart_count}/{max_restarts})")
                await asyncio.sleep(wait_time)
            else:
                logger.critical("Max restarts reached. Stopping.")
                break
    
    logger.info("🏁 Adaptive ML Trading Bot Manager stopped")

if __name__ == "__main__":
    try:
        # Check for required environment variables
        required_vars = ['TELEGRAM_BOT_TOKEN']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            print("Please set these in the Secrets tab")
            sys.exit(1)
        
        # Run the bot manager
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n🛑 Adaptive ML Trading Bot stopped by user")
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        sys.exit(1)
