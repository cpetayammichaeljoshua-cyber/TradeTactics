# Trading Signal Bot

## Overview

This repository contains a comprehensive cryptocurrency trading automation system that processes and forwards trading signals via Telegram. The bot combines multiple trading strategies, machine learning-enhanced analysis, and automated signal forwarding capabilities. It's designed for continuous operation with advanced restart management, health monitoring, and external uptime services integration.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Bot Architecture

The system follows a modular architecture with multiple specialized bot implementations:

- **Signal Processing Layer**: Multiple signal parsers handle various trading signal formats from text messages using regex patterns
- **Strategy Layer**: Advanced trading strategies including Time-Fibonacci theory, ML-enhanced analysis, and multi-timeframe confluence
- **Execution Layer**: Integration with Binance and Kraken exchanges for live trading and market data
- **Communication Layer**: Telegram bot integration for receiving signals and sending formatted responses
- **Persistence Layer**: SQLite database for trade history, user settings, and ML training data

### Trading Strategy Components

The bot implements several sophisticated trading strategies:

1. **Perfect Scalping Strategy**: Uses technical indicators across 3m-4h timeframes with 1:3 risk-reward ratios
2. **Time-Fibonacci Strategy**: Combines market session analysis with Fibonacci retracements for optimal entry timing
3. **ML-Enhanced Strategy**: Machine learning models that learn from past trades to improve future signal quality
4. **Ultimate Scalping Strategy**: Multi-indicator confluence system with dynamic stop-loss management

### Process Management

Robust process management ensures continuous operation:

- **Daemon System**: Auto-restart functionality with configurable retry limits and cooldown periods
- **Health Monitoring**: Memory usage, CPU monitoring, and heartbeat checks
- **Status Tracking**: JSON-based status files for monitoring bot health and performance
- **Keep-Alive Service**: HTTP server for external ping services to maintain Replit uptime

### ML Learning System

Advanced machine learning capabilities for trade optimization:

- **Trade Analysis**: Learns from winning and losing trades to improve signal selection
- **Performance Tracking**: Monitors win rates, profit/loss patterns, and strategy effectiveness
- **Market Insights**: Analyzes optimal trading sessions, symbol performance, and market conditions
- **Adaptive Parameters**: Dynamically adjusts trading parameters based on historical performance

### Risk Management

Comprehensive risk management system:

- **Position Sizing**: Automated calculation based on account balance and risk tolerance
- **Stop-Loss Management**: Dynamic stop-loss adjustment as trades progress through take-profit levels
- **Rate Limiting**: Controlled signal generation to prevent overtrading (2-3 trades per hour)
- **Validation System**: Multi-layer signal validation before execution

## External Dependencies

### Trading Platforms

- **Binance API**: Primary exchange for live trading and market data retrieval
- **Kraken API**: Backup exchange for market data when Binance is unavailable
- **Cornix Integration**: Automated signal forwarding to Cornix trading bots

### Communication Services

- **Telegram Bot API**: Core communication platform for receiving and sending trading signals
- **Telegram Channels**: Signal forwarding to designated channels (@SignalTactics)

### Technical Analysis Libraries

- **pandas-ta**: Technical indicator calculations (RSI, MACD, EMA, Bollinger Bands)
- **talib**: Advanced technical analysis functions when available
- **matplotlib**: Chart generation for signal visualization

### Machine Learning Stack

- **scikit-learn**: Random Forest and Gradient Boosting models for trade prediction
- **pandas/numpy**: Data manipulation and numerical analysis
- **SQLite**: Local database for storing trade history and ML training data

### Infrastructure Services

- **aiohttp**: Asynchronous HTTP client for API communications
- **Flask**: Webhook server for receiving external signals
- **psutil**: Process monitoring and system resource tracking
- **External Ping Services**: Kaffeine, UptimeRobot for maintaining Replit uptime

### Development Libraries

- **asyncio**: Asynchronous programming for concurrent operations
- **logging**: Comprehensive logging system with file rotation
- **signal/threading**: Process management and graceful shutdown handling