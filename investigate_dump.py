import polars as pl
import pickle
from datetime import datetime, timedelta

# Load the tracked results
df = pl.read_csv('signals_2026-01-29_18-08-19_pnl_tracking.csv')

# Filter for completed trades that were selected
selected = df.filter((pl.col('selected') == True) & (pl.col('outcome').is_not_null()))

print('SELECTED TRADES FROM 18:08 BATCH:')
print('='*80)
print(selected.select([
    'pair_address',
    'timestamp',
    'pred_proba',
    'outcome',
    'max_return',
    'min_return',
    'pnl_dollars',
    'capital_invested'
]).sort('pnl_dollars', descending=True))

print('\n\nOUTCOME BREAKDOWN:')
print(selected.group_by('outcome').agg([
    pl.count('outcome').alias('count'),
    pl.sum('pnl_dollars').alias('total_pnl'),
    pl.mean('pnl_dollars').alias('avg_pnl')
]))

# Load cache and check one specific losing trade
cache_file = 'live_candles_cache.pkl'
with open(cache_file, 'rb') as f:
    cache = pickle.load(f)

# Check the worst trade
worst = selected.sort('pnl_dollars').head(1)
if len(worst) > 0:
    pair = worst['pair_address'][0]
    signal_time = datetime.fromisoformat(worst['timestamp'][0])
    
    print(f'\n\nWORST TRADE INVESTIGATION:')
    print(f'Pair: {pair}')
    print(f'Signal Time: {signal_time}')
    print(f'P&L: ${worst["pnl_dollars"][0]:.2f}')
    print(f'Outcome: {worst["outcome"][0]}')
    
    if pair in cache:
        pair_df = cache[pair]
        # Get candles from signal time for next 5 hours
        window = pair_df.filter(
            (pl.col('timestamp') >= signal_time) &
            (pl.col('timestamp') <= signal_time + timedelta(hours=5))
        ).sort('timestamp')
        
        if len(window) > 0:
            print(f'\nPrice Movement (first 10 candles after signal):')
            for i in range(min(10, len(window))):
                time = window[i]['timestamp'][0]
                close = window[i]['close'][0]
                pct = ((close / window[0]['close'][0]) - 1) * 100
                print(f'{time} | Price: {close:.10f} | Change: {pct:+.2f}%')

# Now check a winning trade
best = selected.sort('pnl_dollars', descending=True).head(1)
if len(best) > 0:
    pair = best['pair_address'][0]
    signal_time = datetime.fromisoformat(best['timestamp'][0])
    
    print(f'\n\nBEST TRADE INVESTIGATION:')
    print(f'Pair: {pair}')
    print(f'Signal Time: {signal_time}')
    print(f'P&L: ${best["pnl_dollars"][0]:.2f}')
    print(f'Outcome: {best["outcome"][0]}')
    
    if pair in cache:
        pair_df = cache[pair]
        window = pair_df.filter(
            (pl.col('timestamp') >= signal_time) &
            (pl.col('timestamp') <= signal_time + timedelta(hours=5))
        ).sort('timestamp')
        
        if len(window) > 0:
            print(f'\nPrice Movement (first 10 candles after signal):')
            for i in range(min(10, len(window))):
                time = window[i]['timestamp'][0]
                close = window[i]['close'][0]
                pct = ((close / window[0]['close'][0]) - 1) * 100
                print(f'{time} | Price: {close:.10f} | Change: {pct:+.2f}%')

# Check what percentage of ALL signals hit stop loss
print('\n\n' + '='*80)
print('OVERALL MARKET IMPACT ANALYSIS:')
print('='*80)
all_completed = df.filter(pl.col('outcome').is_not_null())
outcomes = all_completed.group_by('outcome').agg([
    pl.count('outcome').alias('count'),
    (pl.count('outcome') / len(all_completed) * 100).alias('percentage')
])
print(outcomes)

print(f'\nTotal signals: {len(df)}')
print(f'Completed: {len(all_completed)}')
print(f'Still pending: {len(df) - len(all_completed)}')

# Check selected vs non-selected performance
print('\n\nSELECTED vs NON-SELECTED COMPARISON:')
selected_completed = df.filter((pl.col('selected') == True) & (pl.col('outcome').is_not_null()))
non_selected_completed = df.filter((pl.col('selected') == False) & (pl.col('outcome').is_not_null()))

print(f'Selected trades: {len(selected_completed)} completed')
print(f'  Win rate: {(selected_completed.filter(pl.col("outcome") == "win").height / len(selected_completed) * 100):.1f}%')
print(f'  Avg P&L: ${selected_completed["pnl_dollars"].mean():.2f}')

print(f'\nNon-selected trades: {len(non_selected_completed)} completed')
print(f'  Win rate: {(non_selected_completed.filter(pl.col("outcome") == "win").height / len(non_selected_completed) * 100):.1f}%')
if len(non_selected_completed) > 0:
    non_selected_pnl = non_selected_completed['pnl_dollars'].sum()
    print(f'  Total P&L if we had traded all: ${non_selected_pnl:.2f}')
