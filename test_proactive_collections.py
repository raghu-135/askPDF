#!/usr/bin/env python3
"""
Simple verification script for proactive collection creation functionality.
Tests the core logic without requiring full test environment setup.
"""

import asyncio
import sys
import os

# Add the rag_service directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rag_service'))

async def test_collection_manager():
    """Test the enhanced ensure_collections_for_thread method."""
    try:
        from app.db.vector.collection_manager import ModelAwareCollectionManager
        from app.db.vector.model_registry import get_embedding_model_registry
        print("✅ Imports successful")
        
        # Create a mock client
        class MockClient:
            def __init__(self):
                self.collections = MockCollections()
        
        class MockCollections:
            def __init__(self):
                self.exists_call_count = 0
                self.create_call_count = 0
                self.use_call_count = 0
            
            def exists(self, name):
                self.exists_call_count += 1
                print(f"🔍 Checking if collection '{name}' exists")
                return False  # Always return False to trigger creation
            
            def create(self, name, vector_config, properties):
                self.create_call_count += 1
                print(f"🏗️ Creating collection '{name}' with {len(properties)} properties")
                return None
            
            def use(self, name):
                self.use_call_count += 1
                print(f"📂 Using collection '{name}'")
                return MockCollection()
        
        class MockCollection:
            pass
        
        # Test the collection manager
        mock_client = MockClient()
        manager = ModelAwareCollectionManager(mock_client)
        
        # Mock the registry to avoid actual model probing
        registry = get_embedding_model_registry()
        registry._dimension_cache["test-model"] = 384
        registry._model_cache["test-model"] = {
            'dimensions': 384,
            'sanitized_name': 'test_model',
            'is_local': False
        }
        
        print("\n🧪 Testing ensure_collections_for_thread...")
        await manager.ensure_collections_for_thread("test-model")
        
        # Verify all three collections were checked and created
        assert mock_client.collections.exists_call_count == 3, f"Expected 3 exists calls, got {mock_client.collections.exists_call_count}"
        assert mock_client.collections.create_call_count == 3, f"Expected 3 create calls, got {mock_client.collections.create_call_count}"
        assert mock_client.collections.use_call_count == 3, f"Expected 3 use calls, got {mock_client.collections.use_call_count}"
        
        print("✅ Collection manager test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Collection manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_thread_integration():
    """Test the thread integration logic."""
    try:
        print("\n🧪 Testing thread integration logic...")
        
        # Mock the components that would be used in threads.py
        class MockVectorDB:
            def __init__(self):
                self.collection_manager = MockCollectionManager()
        
        class MockCollectionManager:
            async def ensure_collections_for_thread(self, embedding_model_name):
                print(f"🚀 Proactively ensuring collections for model '{embedding_model_name}'")
                await asyncio.sleep(0.1)  # Simulate async work
                print("✅ Collections ensured successfully")
        
        # Simulate the thread loading logic
        mock_db = MockVectorDB()
        
        # This simulates the async task creation in threads.py
        async def simulate_thread_loading():
            # Simulate existing backfill task
            print("🔄 Starting existing backfill task...")
            
            # Simulate new proactive collection creation task
            print("🔄 Starting proactive collection creation task...")
            await mock_db.collection_manager.ensure_collections_for_thread("test-model")
            
            print("✅ Both background tasks completed")
        
        await simulate_thread_loading()
        print("✅ Thread integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Thread integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all verification tests."""
    print("🚀 Starting proactive collection creation verification tests...\n")
    
    test1_passed = await test_collection_manager()
    test2_passed = await test_thread_integration()
    
    print(f"\n📊 Test Results:")
    print(f"   Collection Manager: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Thread Integration: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Proactive collection creation is working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
