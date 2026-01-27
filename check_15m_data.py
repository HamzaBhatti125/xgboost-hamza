import pandas as pd
import pyarrow.parquet as pq

# Read parquet metadata without loading full file
parquet_file = pq.ParquetFile('Files/candles-15m.parquet')

print('File metadata:')
print(f'Number of rows: {parquet_file.metadata.num_rows:,}')
print(f'Number of columns: {parquet_file.metadata.num_columns}')
print(f'\nColumns: {parquet_file.schema.names}')
print(f'\nSchema:')
print(parquet_file.schema)

# Read first chunk to see data structure
print('\n' + '='*50)
print('Reading first 1000 rows...')
print('='*50)

# Use PyArrow to read specific rows
table = parquet_file.read_row_group(0, columns=parquet_file.schema.names)
df_sample = table.to_pandas().head(1000)

print('\nFirst few rows:')
print(df_sample.head(10))

print('\nData types:')
print(df_sample.dtypes)

print('\nBasic stats (first 1000 rows):')
print(df_sample.describe())

print('\nNull values (first 1000 rows):')
print(df_sample.isnull().sum())

# Memory usage estimate
print(f'\nMemory usage for sample: {df_sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
print(f'Estimated total file size: {(df_sample.memory_usage(deep=True).sum() / 1024**2) * (parquet_file.metadata.num_rows / 1000):.2f} MB')
