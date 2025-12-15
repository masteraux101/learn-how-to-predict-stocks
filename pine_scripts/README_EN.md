# Candle Pattern Strategy - Pine Script V6

## Overview

A TradingView trading strategy script (Pine Script V6) that identifies candlestick patterns over a configurable lookback period and generates trading signals based on pattern recognition rules.

## Strategy Logic

The strategy analyzes the most recent N candlesticks (default: 5) and classifies their patterns into three categories, then generates buy/sell signals based on specific pattern combinations.

### Candlestick Pattern Classification

Each candlestick is classified as one of three types:

| Pattern | Definition | Formula |
|---------|-----------|---------|
| **Close at High** | Close price near the highest price of the candle | `(high - close) / (high - low) ≤ (1 - threshold)` |
| **Close at Low** | Close price near the lowest price of the candle | `(close - low) / (high - low) ≤ (1 - threshold)` |
| **Doji** | Opening and closing prices very close; long upper/lower wicks | `abs(close - open) / (high - low) ≤ threshold` |

### Trading Rules

| Condition | Action | Signal |
|-----------|--------|--------|
| Number of Doji candles ≥ 3 | **SKIP** - Market is too chaotic | No trade |
| "Close at High" count ≥ 2 + Last candle closes at high | **LONG** - Bullish confirmation | Buy Signal |
| "Close at Low" count ≥ 2 + Last candle closes at low | **SHORT** - Bearish confirmation | Sell Signal |

## Configuration Parameters

All parameters are adjustable in TradingView's "Settings/Inputs" tab:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **K-line Lookback Period** | 5 | 3-20 | Number of candlesticks to analyze |
| **Doji Threshold** | 0.1 | 0.0-1.0 | Body size / Total range ratio for doji identification |
| **Doji Count Limit** | 3 | 1-10 | Number of dojis that triggers skip signal |
| **Close at High Proximity** | 0.9 | 0.5-1.0 | Required proximity to highest price (0.9 = 90%) |
| **Close at Low Proximity** | 0.9 | 0.5-1.0 | Required proximity to lowest price (0.9 = 90%) |

### Parameter Tuning Guide

#### Proximity Thresholds (Close at High/Low)
- **0.95 or higher**: Stricter - only very strong candles qualify
- **0.90 (default)**: Balanced - good for most market conditions
- **0.80 or lower**: Looser - catches more patterns, higher false signals

#### Doji Threshold
- **0.05-0.10**: Strict doji identification
- **0.10-0.15**: Standard doji detection
- **0.15+**: Includes near-doji patterns

#### Lookback Period
- **3-5 bars**: Fast, sensitive to recent patterns
- **5-10 bars**: Balanced approach
- **10-20 bars**: Slower, filters out noise

## Visual Indicators

### Chart Markers
- **Green Up Arrow** ▲: Long entry signal triggered
- **Red Down Arrow** ▼: Short entry signal triggered
- **Gray X**: Skip signal (too many dojis detected)

### Statistics Table
Located at top-right corner, displays real-time pattern counts:
- **Close at High**: Count of candles closing near the high
- **Close at Low**: Count of candles closing near the low
- **Doji**: Count of doji/near-doji patterns

### Signal Labels
- Green label: "✓ Long condition met" - below entry bar
- Red label: "✓ Short condition met" - above entry bar
- Gray label: "⊠ Too many dojis, skip" - trade skipped

## Risk Management

### Exit Strategy
The strategy includes built-in exit rules based on ATR (Average True Range):

- **Stop Loss**: ATR × 2
- **Take Profit**: ATR × 3

Example: If ATR is $10, stop loss is at -$20 and take profit is at +$30

### Position Sizing
Default: 100% of equity per trade (adjustable in strategy properties)

**Recommendation**: Reduce to 10-25% of equity for safer capital allocation

## How to Use

### 1. Copy the Script

1. Open TradingView
2. Go to Pine Script Editor
3. Create a new indicator
4. Copy the entire `candle_pattern_strategy.pine` content
5. Click "Add to Chart"

### 2. Apply to Chart

- Select your preferred timeframe (1m, 5m, 15m, 1h, 4h, 1d)
- Choose a trading symbol (stocks, forex, crypto, etc.)
- The strategy will generate signals on all historical bars and in real-time

### 3. Configure Parameters

1. Right-click the strategy on chart → "Settings"
2. Adjust parameters in the "Inputs" tab
3. Strategy will recalculate automatically
4. Backtest results appear in the Strategy Tester panel

### 4. Backtesting

1. Click "Strategy Tester" (bottom panel)
2. Set date range for historical testing
3. Review performance metrics:
   - Win rate
   - Profit factor
   - Drawdown
   - Total returns

## Key Features

✅ **Execution Model Compliant** - Follows official Pine Script V6 standards
✅ **Global Scope** - All historical references in global scope for consistency
✅ **Adjustable Sensitivity** - Fine-tune all pattern detection thresholds
✅ **Real-time Updates** - Works on both historical and real-time bars
✅ **Visual Feedback** - Clear signals with labels and statistics table
✅ **Risk Management** - Built-in ATR-based stop loss and take profit
✅ **No Repainting** - Signals remain consistent after bar closes

## Important Notes

### Before Trading Live

1. **Backtest thoroughly**: Test on at least 6-12 months of historical data
2. **Paper trading first**: Practice with simulated trades
3. **Optimize parameters**: Find the best settings for your trading instrument
4. **Manage risk**: Never risk more than 1-2% per trade
5. **Monitor performance**: Review trades regularly and adjust as needed

### Market Conditions

The strategy works best in:
- Trending markets with clear directional bias
- Range-bound markets with defined support/resistance
- Markets with moderate volatility

The strategy may underperform in:
- Highly choppy/noisy markets (many false signals)
- Markets with gaps or overnight gaps
- Low liquidity periods

### Limitations

- Works on chart timeframes only (not intrabar analysis)
- Requires at least 5 bars of history (configurable)
- Fixed exit rules may not suit all trading styles
- No trend filtering (will trade both directions)

## Technical Details

### Historical Buffer Management

The strategy declares maximum bars back to ensure enough historical data:
```pine
max_bars_back(close, 20)
max_bars_back(open, 20)
max_bars_back(high, 20)
max_bars_back(low, 20)
```

This prevents runtime errors when accessing historical candlestick data.

### Pattern Detection Algorithm

```
For each bar in lookback period:
  1. Calculate (high - close) / (high - low) → close_to_high_ratio
  2. If close_to_high_ratio ≤ (1 - threshold) → count as "Close at High"
  
  3. Calculate (close - low) / (high - low) → close_to_low_ratio
  4. If close_to_low_ratio ≤ (1 - threshold) → count as "Close at Low"
  
  5. Calculate abs(close - open) / (high - low) → body_ratio
  6. If body_ratio ≤ doji_threshold → count as "Doji"

Signal Generation:
  7. If doji_count ≥ cross_count_limit → SKIP_SIGNAL
  8. If close_at_high_count ≥ 2 AND last_close_at_high → LONG_SIGNAL
  9. If close_at_low_count ≥ 2 AND last_close_at_low → SHORT_SIGNAL
```

## Example Scenarios

### Scenario 1: Bullish Signal
```
Last 5 bars pattern:
Bar 1: Closes at high (95%)  ✓
Bar 2: Closes at high (92%)  ✓
Bar 3: Closes at mid (50%)
Bar 4: Closes at low (10%)
Bar 5: Closes at high (96%)  ✓ (last bar)

Result:
- Close at high count: 3 (≥ 2) ✓
- Last closes at high: ✓
- Doji count: 0 (< 3) ✓
→ LONG SIGNAL ✓
```

### Scenario 2: Skip Signal
```
Last 5 bars pattern:
Bar 1: Doji pattern      ✓
Bar 2: Doji pattern      ✓
Bar 3: Closes at mid
Bar 4: Doji pattern      ✓
Bar 5: Closes at high

Result:
- Doji count: 3 (≥ 3)
→ SKIP SIGNAL - Market too chaotic
```

## Performance Optimization

### Tips for Better Results

1. **Use Higher Timeframes**: 15m or higher reduces false signals
2. **Combine with Indicators**: Add volume or trend filters
3. **Seasonal Adjustment**: Different parameters for different seasons
4. **Instrument Selection**: Test on liquid, high-volume instruments
5. **Regular Rebalancing**: Adjust parameters quarterly based on performance

## Troubleshooting

### No Signals Generated
- Check if lookback period is too large
- Verify proximity thresholds aren't too strict (try 0.85 instead of 0.95)
- Ensure doji limit allows for trading (try 4-5 instead of 3)

### Too Many False Signals
- Increase proximity thresholds to 0.95+
- Reduce lookback period to 3-4 bars
- Increase doji limit to 3-4
- Test on higher timeframes

### Strategy Tester Shows Losses
- Review individual trades to identify patterns
- Adjust stop loss and take profit ratios
- Change lookback period and thresholds
- Test on different market conditions

## References

- [Pine Script V6 Documentation](https://www.tradingview.com/pine-script-docs/)
- [Execution Model Guide](https://www.tradingview.com/pine-script-docs/language/execution-model/)
- [Candlestick Patterns Reference](https://www.investopedia.com/terms/c/candlestick.asp)
- [ATR Indicator Guide](https://www.investopedia.com/terms/a/atr.asp)

## Disclaimer

**This strategy is provided for educational and research purposes only.**

- Not financial advice
- Past performance does not guarantee future results
- Always conduct thorough backtesting before live trading
- Trading involves significant risk of loss
- Use proper risk management and position sizing

## License

Created: December 15, 2025
Version: 1.0

---

**For updates and modifications, refer to the accompanying documentation files.**
