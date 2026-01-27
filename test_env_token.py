#!/usr/bin/env python3
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv('ENVIO_API_TOKEN')

if token:
    print('✅ ENVIO_API_TOKEN loaded successfully!')
    print(f'   Token length: {len(token)} characters')
    print(f'   First 10 chars: {token[:10]}...')
else:
    print('❌ ENVIO_API_TOKEN not found in .env file')
    print('\nMake sure your .env file contains:')
    print('ENVIO_API_TOKEN=your_token_here')
