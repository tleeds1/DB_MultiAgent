#!/usr/bin/env python3
"""
Test script for the enhanced notification system
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.notification import NotificationManager

def test_notification():
    """Test the enhanced notification system"""
    print("🧪 Testing Enhanced Notification System...")
    
    # Create notification manager
    notification_manager = NotificationManager()
    
    # Sample query results
    sample_results = {
        'rows': [
            ['tleeds1', '2604200520', 'None', 'VN', '82'],
            ['ledoantho', '2302200520', 'None', 'VN', '82']
        ],
        'columns': ['customer', 'addressid', 'addressrepresentationcode', 'country', 'region'],
        'row_count': 2
    }
    
    # Sample context
    sample_context = {
        'language': 'vi',
        'complexity': 'simple',
        'operation_type': 'read',
        'mentioned_tables': ['customers', 'addresses'],
        'requires_join': True
    }
    
    # Create structured notification
    notification = notification_manager.create_structured_notification(
        question="Tìm cho tôi người dùng ở nước VN vùng 82",
        sql="SELECT c.customer, a.addressid, a.addressrepresentationcode, a.country, a.region FROM customers c JOIN addresses a ON c.addressid = a.addressid WHERE a.country = 'VN' AND a.region = '82'",
        results=sample_results,
        execution_time=2.19,
        context=sample_context,
        status="SUCCESS"
    )
    
    print("\n📱 Generated Notification:")
    print("=" * 60)
    print(notification)
    print("=" * 60)
    
    print("\n✅ Notification test completed!")
    print("📋 The notification now includes:")
    print("  - User question")
    print("  - Generated SQL query")
    print("  - Query results (up to 5 rows)")
    print("  - Performance analysis")
    print("  - Context information")

if __name__ == "__main__":
    test_notification()
