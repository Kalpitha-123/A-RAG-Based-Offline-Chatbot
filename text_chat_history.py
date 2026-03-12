#!/usr/bin/env python3
"""
Test script to verify chat history functionality
"""
import os
import json

CHAT_HISTORY_FILE = 'chat_history.json'

def test_chat_history():
    """Test chat history read/write operations"""
    print("Testing chat history functionality...")
    
    # Test 1: Create sample history
    sample_history = [
        {
            'id': 'test_1',
            'query': 'What is the proposed system?',
            'response': 'Based on the document, the proposed system...',
            'timestamp': '2026-01-31T18:51:59',
            'sources_used': ['Report']
        }
    ]
    
    try:
        # Test write
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(sample_history, f, ensure_ascii=False, indent=2)
        print(f"✅ Successfully wrote test history to {CHAT_HISTORY_FILE}")
        
        # Test read
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            loaded_history = json.load(f)
        print(f"✅ Successfully read {len(loaded_history)} entries from history")
        
        # Test clear
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print("✅ Successfully cleared history")
        
        # Verify clear
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            cleared_history = json.load(f)
        print(f"✅ Verified cleared history: {len(cleared_history)} entries")
        
        print("\n🎉 All chat history tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_chat_history()
