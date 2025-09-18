
#!/usr/bin/env python3
"""
Ultimate Trading Bot Launcher
Optimized launcher with auto-restart and process management
"""

import os
import sys
import asyncio
import signal
from pathlib import Path

# Add SignalMaestro to path
sys.path.insert(0, str(Path(__file__).parent / "SignalMaestro"))

from ultimate_trading_bot import main

def main_launcher():
    """Main launcher with auto-restart capability"""
    restart_count = 0
    max_restarts = 100
    
    print("🚀 Ultimate Trading Bot Launcher")
    print("🔧 Optimized for maximum profitability")
    print("🌐 Starting with auto-restart protection...")
    
    # Check for required environment variables
    required_vars = ['TELEGRAM_BOT_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these in the Secrets tab in Replit")
        return
    
    while restart_count < max_restarts:
        try:
            print(f"\n🎯 Starting Ultimate Trading Bot (attempt #{restart_count + 1})")
            
            # Run the bot
            should_restart = asyncio.run(main())
            
            if not should_restart:
                print("🛑 Bot stopped manually")
                break
            
            restart_count += 1
            print(f"🔄 Auto-restart #{restart_count}/{max_restarts} in 15 seconds...")
            
            # Progressive restart delay
            if restart_count <= 5:
                delay = 15
            elif restart_count <= 10:
                delay = 30
            else:
                delay = 60
            
            print(f"⏳ Waiting {delay} seconds before restart...")
            import time
            time.sleep(delay)
            
        except KeyboardInterrupt:
            print("\n🛑 Manual shutdown requested")
            break
        except Exception as e:
            restart_count += 1
            print(f"💥 Critical error #{restart_count}: {e}")
            print(f"🔄 Restarting in 30 seconds...")
            import time
            time.sleep(30)
    
    if restart_count >= max_restarts:
        print(f"⚠️ Maximum restart limit reached ({max_restarts})")
    
    print("✅ Ultimate Trading Bot launcher shutdown complete")

if __name__ == "__main__":
    main_launcher()
