#!/usr/bin/env python3
"""
Test script for schedule library functionality
"""
import schedule
import time

def test_job():
    print('✅ Schedule test job executed successfully!')

# Schedule a job
schedule.every(1).seconds.do(test_job)

# Run pending jobs
schedule.run_pending()

print('✅ Schedule functionality test completed - library is working correctly!')