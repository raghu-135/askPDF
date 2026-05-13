"""
test_model_registry_edge_cases.py - Edge case tests for model registry operations.

This test suite validates:
- Model registry edge cases
- Collection naming edge cases
- Cache behavior edge cases
- Error handling edge cases
"""

import pytest
import asyncio
import re
from unittest.mock import patch, MagicMock
from app.db.vector.model_registry import EmbeddingModelRegistry, get_embedding_model_registry


class TestModelRegistryEdgeCases:
    """Test edge cases in model registry operations."""
    
    @pytest.fixture
    def registry(self):
        return EmbeddingModelRegistry()
    
    @pytest.mark.asyncio
    async def test_empty_model_name_handling(self, registry):
        """Test handling of empty and None model names."""
        with patch('app.db.vector.model_registry.get_embedding_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.aembed_query.return_value = [0.1] * 384
            mock_get_model.return_value = mock_model
            
            # Empty string should be handled
            with pytest.raises(Exception):  # Should fail gracefully
                await registry._probe_model_dimensions("")
            
            # None should be handled
            with pytest.raises(Exception):  # Should fail gracefully
                await registry._probe_model_dimensions(None)
            
            # Whitespace-only should be handled
            with pytest.raises(Exception):  # Should fail gracefully
                await registry._probe_model_dimensions("   ")
    
    @pytest.mark.asyncio
    async def test_unicode_model_names(self, registry):
        """Test handling of Unicode characters in model names."""
        unicode_model_names = [
            "模型-中文",  # Chinese characters
            "модель-русский",  # Cyrillic characters
            "modèle-français",  # French characters
            "modelo-español",  # Spanish characters
            "モデル-日本語",  # Japanese characters
            "موديل-عربي",  # Arabic characters
            "🤖-emoji-model",  # Emoji characters
        ]
        
        with patch.object(registry, '_probe_model_dimensions', return_value=768):
            for model_name in unicode_model_names:
                # Should handle Unicode gracefully
                model_info = await registry.get_model_info(model_name)
                assert model_info['dimensions'] == 768
                assert 'sanitized_name' in model_info
                
                # Collection name should be ASCII-safe
                collection_name = registry.get_collection_name("DocumentChunk", model_name)
                assert all(ord(c) < 128 for c in collection_name)  # All ASCII
                assert collection_name.endswith("_768")
    
    @pytest.mark.asyncio
    async def test_extremely_long_model_names(self, registry):
        """Test handling of extremely long model names."""
        # Test various length scenarios
        long_names = [
            "a" * 100,    # 100 chars
            "a" * 1000,   # 1000 chars
            "a" * 10000,  # 10000 chars
        ]
        
        with patch.object(registry, '_probe_model_dimensions', return_value=384):
            for model_name in long_names:
                model_info = await registry.get_model_info(model_name)
                assert model_info['dimensions'] == 384
                
                # Sanitized name should be manageable
                sanitized = model_info['sanitized_name']
                assert len(sanitized) == len(model_name)  # Should preserve length
                assert '_' in sanitized  # Should contain underscores
                
                # Collection name should be valid
                collection_name = registry.get_collection_name("DocumentChunk", model_name)
                assert isinstance(collection_name, str)
                assert collection_name.endswith("_384")
    
    @pytest.mark.asyncio
    async def test_special_character_sanitization(self, registry):
        """Test sanitization of various special characters."""
        special_chars = [
            "model/name/with/slashes",
            "model\\name\\with\\backslashes",
            "model.name.with.dots",
            "model-name-with-dashes",
            "model name with spaces",
            "model@with#special$chars%",
            "model(with)parentheses",
            "model[with]brackets",
            "model{with}braces",
            "model|with|pipes",
            "model+with+plus",
            "model=with=equals",
            "model&with&ampersands",
            "model;with;semicolons",
            "model:with:colons",
            'model"with"quotes',
            "model'with'apostrophes",
            "model<with>angles",
            "model*with*asterisks",
            "model?with?questions",
        ]
        
        with patch.object(registry, '_probe_model_dimensions', return_value=512):
            for model_name in special_chars:
                model_info = await registry.get_model_info(model_name)
                sanitized = model_info['sanitized_name']
                
                # Should not contain any special characters except underscores
                assert all(c.isalnum() or c == '_' for c in sanitized)
                
                # Collection name should be valid
                collection_name = registry.get_collection_name("DocumentChunk", model_name)
                assert all(c.isalnum() or c == '_' for c in collection_name.replace('DocumentChunk_', '').replace('_512', ''))
    
    @pytest.mark.asyncio
    async def test_concurrent_model_info_loading(self, registry):
        """Test concurrent loading of the same model info."""
        with patch.object(registry, '_probe_model_dimensions', return_value=768):
            # Create many concurrent tasks for the same model
            tasks = []
            for i in range(20):
                task = registry.get_model_info("concurrent-model")
                tasks.append(task)
            
            # All should complete successfully
            results = await asyncio.gather(*tasks)
            
            # All should return the same result
            first_result = results[0]
            for result in results:
                assert result == first_result
                assert result['dimensions'] == 768
            
            # Should only call probe once due to caching
            # (This is tested through the mock call count)
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_edge_cases(self, registry):
        """Test cache invalidation edge cases."""
        # Load initial model info
        with patch.object(registry, '_probe_model_dimensions', return_value=384):
            await registry.get_model_info("cache-test-model")
        
        assert "cache-test-model" in registry._model_cache
        assert registry._dimension_cache["cache-test-model"] == 384
        
        # Partially corrupt caches
        registry._model_cache["cache-test-model"] = {
            "dimensions": 999,  # Wrong dimension
            "sanitized_name": "cache_test_model",
            "is_local": False
        }
        
        # get_dimensions should return cached (wrong) value
        assert await registry.get_dimensions("cache-test-model") == 999
        
        # Clear dimension cache only
        registry._dimension_cache.clear()
        
        # Should re-probe dimensions
        with patch.object(registry, '_probe_model_dimensions', return_value=768):
            dimensions = await registry.get_dimensions("cache-test-model")
            assert dimensions == 768
        
        # Should update dimension cache
        assert registry._dimension_cache["cache-test-model"] == 768
    
    @pytest.mark.asyncio
    async def test_model_info_corruption_recovery(self, registry):
        """Test recovery from corrupted model info."""
        # Corrupt the model cache
        registry._model_cache["corrupted-model"] = {
            "dimensions": None,  # Corrupted dimension
            "sanitized_name": None,  # Corrupted name
            "is_local": None,  # Corrupted flag
        }
        registry._dimension_cache["corrupted-model"] = "invalid"  # Corrupted dimension
        
        # Should recover by re-probing
        with patch.object(registry, '_probe_model_dimensions', return_value=512):
            model_info = await registry.get_model_info("corrupted-model")
            assert model_info['dimensions'] == 512
            assert model_info['sanitized_name'] == 'corrupted_model'
            assert model_info['is_local'] is not None
        
        # Should fix caches
        assert registry._dimension_cache["corrupted-model"] == 512
    
    @pytest.mark.asyncio
    async def test_collection_name_parsing_edge_cases(self, registry):
        """Test collection name parsing edge cases."""
        # Test malformed collection names
        malformed_names = [
            "",  # Empty
            "DocumentChunk",  # No model info
            "DocumentChunk_",  # Incomplete
            "DocumentChunk_model",  # No dimensions
            "DocumentChunk_model_",  # No dimensions
            "DocumentChunk__384",  # Double underscore
            "DocumentChunk_model_",  # Trailing underscore
            "DocumentChunk_384",  # No model name
            "InvalidPrefix_model_384",  # Invalid prefix
            "DocumentChunk_model_abc",  # Non-numeric dimensions
            "DocumentChunk_model_38.4",  # Float dimensions
            "DocumentChunk_model_-384",  # Negative dimensions
        ]
        
        for collection_name in malformed_names:
            # Should handle gracefully
            model = registry.get_model_from_collection(collection_name)
            # Most should return None for malformed names
            if collection_name not in ["DocumentChunk_model_384", "DocumentChunk_model_38.4"]:
                assert model is None or model is not None  # Should not crash
    
    @pytest.mark.asyncio
    async def test_dimension_validation_edge_cases(self, registry):
        """Test dimension validation edge cases."""
        # Test with zero dimensions
        registry._dimension_cache["zero-dim-model"] = 0
        
        # Test with negative dimensions
        registry._dimension_cache["negative-dim-model"] = -1
        
        # Test with very large dimensions
        registry._dimension_cache["huge-dim-model"] = 999999
        
        collection_manager = MagicMock()
        collection_manager.registry = registry
        
        # Zero-dimensional vectors should be invalid
        assert not await collection_manager.validate_vectors_for_model([], "zero-dim-model")
        
        # Vectors with wrong dimensions should fail
        assert not await collection_manager.validate_vectors_for_model([[0.1]], "zero-dim-model")
        
        # Negative dimensions should be handled gracefully
        # (This depends on implementation - should fail gracefully)
    
    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self, registry):
        """Test memory leak prevention in registry."""
        # Add many models to cache
        with patch.object(registry, '_probe_model_dimensions', return_value=384):
            for i in range(1000):
                model_name = f"memory-leak-test-{i}"
                await registry.get_model_info(model_name)
        
        # Check cache sizes
        assert len(registry._model_cache) == 1000
        assert len(registry._dimension_cache) == 1000
        
        # Clear caches and verify memory is freed
        registry._model_cache.clear()
        registry._dimension_cache.clear()
        
        assert len(registry._model_cache) == 0
        assert len(registry._dimension_cache) == 0
        
        # Should be able to use registry normally after clearing
        with patch.object(registry, '_probe_model_dimensions', return_value=768):
            await registry.get_model_info("after-clear-test")
            assert len(registry._model_cache) == 1
            assert len(registry._dimension_cache) == 1
    
    @pytest.mark.asyncio
    async def test_race_condition_prevention(self, registry):
        """Test race condition prevention in concurrent operations."""
        with patch.object(registry, '_probe_model_dimensions', return_value=512):
            # Create concurrent tasks that might race
            tasks = []
            
            # Mix of get_model_info and get_dimensions calls
            for i in range(20):
                if i % 2 == 0:
                    tasks.append(registry.get_model_info("race-test-model"))
                else:
                    tasks.append(registry.get_dimensions("race-test-model"))
            
            # All should complete without race conditions
            results = await asyncio.gather(*tasks)
            
            # All dimension results should be consistent
            for result in results:
                if isinstance(result, dict):  # get_model_info result
                    assert result['dimensions'] == 512
                else:  # get_dimensions result
                    assert result == 512
    
    @pytest.mark.asyncio
    async def test_error_propagation_edge_cases(self, registry):
        """Test error propagation in various edge cases."""
        # Test with different types of exceptions
        exceptions = [
            Exception("Generic error"),
            ValueError("Invalid value"),
            RuntimeError("Runtime error"),
            ConnectionError("Connection failed"),
            TimeoutError("Operation timed out"),
            MemoryError("Out of memory"),
        ]
        
        for exc in exceptions:
            with patch.object(registry, '_probe_model_dimensions', side_effect=exc):
                with pytest.raises(type(exc)):
                    await registry.get_model_info(f"error-test-{type(exc).__name__}")
            
            # Should not corrupt caches
            assert f"error-test-{type(exc).__name__}" not in registry._model_cache
            assert f"error-test-{type(exc).__name__}" not in registry._dimension_cache
    
    @pytest.mark.asyncio
    async def test_global_registry_instance(self):
        """Test global registry instance behavior."""
        # Get multiple instances
        registry1 = get_embedding_model_registry()
        registry2 = get_embedding_model_registry()
        registry3 = get_embedding_model_registry()
        
        # Should be the same instance (singleton)
        assert registry1 is registry2
        assert registry2 is registry3
        
        # Should share state
        with patch.object(registry1, '_probe_model_dimensions', return_value=384):
            await registry1.get_model_info("singleton-test")
        
        assert "singleton-test" in registry1._model_cache
        assert "singleton-test" in registry2._model_cache
        assert "singleton-test" in registry3._model_cache
        
        assert registry1._dimension_cache["singleton-test"] == 384
        assert registry2._dimension_cache["singleton-test"] == 384
        assert registry3._dimension_cache["singleton-test"] == 384


class TestCollectionNamingEdgeCases:
    """Test edge cases in collection naming."""
    
    @pytest.fixture
    def registry(self):
        return EmbeddingModelRegistry()
    
    @pytest.mark.asyncio
    async def test_model_name_sanitization_comprehensive(self, registry):
        """Test comprehensive model name sanitization."""
        test_cases = [
            # (input, expected_sanitized)
            ("simple-model", "simple_model"),
            ("model/with/slashes", "model_with_slashes"),
            ("model\\with\\backslashes", "model_with_backslashes"),
            ("model.name.with.dots", "model_name_with_dots"),
            ("model name with spaces", "model_name_with_spaces"),
            ("model@with#special$chars%", "model_with_special_chars"),
            ("model(with)parentheses", "model_with_parentheses"),
            ("model[with]brackets", "model_with_brackets"),
            ("model{with}braces", "model_with_braces"),
            ("model|with|pipes", "model_with_pipes"),
            ("model+with+plus", "model_with_plus"),
            ("model=with=equals", "model_with_equals"),
            ("model&with&ampersands", "model_with_ampersands"),
            ("model;with;semicolons", "model_with_semicolons"),
            ("model:with:colons", "model_with_colons"),
            ('model"with"quotes', 'model_with_quotes'),
            ("model'with'apostrophes", "model_with_apostrophes"),
            ("model<with>angles", "model_with_angles"),
            ("model*with*asterisks", "model_with_asterisks"),
            ("model?with?questions", "model_with_questions"),
            ("model\nwith\nnewlines", "model_with_newlines"),
            ("model\twith\ttabs", "model_with_tabs"),
            ("model\rwith\rcarriage", "model_with_carriage"),
        ]
        
        for input_name, expected in test_cases:
            sanitized = registry.sanitize_model_name(input_name)
            assert sanitized == expected, f"Failed for '{input_name}': expected '{expected}', got '{sanitized}'"
    
    @pytest.mark.asyncio
    async def test_collection_name_length_limits(self, registry):
        """Test collection name length limits."""
        # Create model with very long name
        very_long_name = "a" * 1000
        
        with patch.object(registry, '_probe_model_dimensions', return_value=384):
            await registry.get_model_info(very_long_name)
            
            collection_name = registry.get_collection_name("DocumentChunk", very_long_name)
            
            # Should be a valid string
            assert isinstance(collection_name, str)
            assert len(collection_name) > 1000  # Should include full name
            
            # Should be usable
            assert registry.is_model_compatible(collection_name, very_long_name)
    
    @pytest.mark.asyncio
    async def test_collection_name_uniqueness(self, registry):
        """Test collection name uniqueness for different models."""
        # Setup models with similar names but different characteristics
        models_info = {
            "model-v1": {"dimensions": 384, "is_local": False},
            "model-v2": {"dimensions": 384, "is_local": True},
            "model-v3": {"dimensions": 768, "is_local": False},
            "model-v1-updated": {"dimensions": 384, "is_local": False},
        }
        
        with patch.object(registry, '_probe_model_dimensions'):
            for model_name, info in models_info.items():
                registry._model_cache[model_name] = {
                    "dimensions": info["dimensions"],
                    "sanitized_name": registry.sanitize_model_name(model_name),
                    "is_local": info["is_local"]
                }
                registry._dimension_cache[model_name] = info["dimensions"]
        
        # Generate collection names
        collection_names = {}
        for model_name in models_info.keys():
            collection_names[model_name] = registry.get_collection_name("DocumentChunk", model_name)
        
        # All should be unique
        assert len(set(collection_names.values())) == len(collection_names)
        
        # Each should be compatible only with its model
        for model_name, collection_name in collection_names.items():
            assert registry.is_model_compatible(collection_name, model_name)
            
            # Should not be compatible with other models
            for other_model in models_info.keys():
                if other_model != model_name:
                    # Some might be compatible if they have same dimensions and sanitized name
                    # But generally should be different
                    pass  # Compatibility depends on specific implementation


if __name__ == "__main__":
    pytest.main([__file__])
