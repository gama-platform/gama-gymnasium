"""
Performance tests for GAMA-Gymnasium.

These tests verify that the system meets performance requirements
and identify potential bottlenecks.
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from gama_gymnasium import GamaEnv, SyncWrapper
from gama_gymnasium.spaces.converters import map_to_space


class TestGamaGymnasiumPerformance:
    """Performance tests for GAMA-Gymnasium components."""
    
    @pytest.fixture
    def fast_mock_responses(self):
        """Mock responses optimized for performance testing."""
        return {
            "load": {"type": "CommandExecutedSuccessfully", "content": "experiment_123"},
            "observation_space": {"type": "CommandExecutedSuccessfully", "content": {"type": "Discrete", "n": 4}},
            "action_space": {"type": "CommandExecutedSuccessfully", "content": {"type": "Discrete", "n": 2}},
            "reset_state": {"type": "CommandExecutedSuccessfully", "content": 0},
            "reset_info": {"type": "CommandExecutedSuccessfully", "content": {}},
            "step_data": {"type": "CommandExecutedSuccessfully", "content": {
                "State": 1, "Reward": 1.0, "Terminated": False, "Truncated": False, "Info": {}
            }}
        }
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    @patch('gama_gymnasium.core.client.MessageTypes')
    def test_environment_creation_performance(self, mock_message_types, mock_sync_client, fast_mock_responses):
        """Test environment creation time."""
        mock_message_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
        mock_client = mock_sync_client.return_value
        mock_client.load.return_value = fast_mock_responses["load"]
        mock_client.expression.side_effect = [
            fast_mock_responses["observation_space"],
            fast_mock_responses["action_space"]
        ]
        
        # Measure creation time
        start_time = time.time()
        env = GamaEnv(
            gaml_experiment_path="test.gaml",
            gaml_experiment_name="test_experiment"
        )
        creation_time = time.time() - start_time
        
        # Should create environment quickly (< 1 second)
        assert creation_time < 1.0
        
        env.close()
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    @patch('gama_gymnasium.core.client.MessageTypes')
    def test_step_performance(self, mock_message_types, mock_sync_client, fast_mock_responses):
        """Test step execution performance."""
        mock_message_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
        mock_client = mock_sync_client.return_value
        mock_client.load.return_value = fast_mock_responses["load"]
        mock_client.expression.side_effect = [
            fast_mock_responses["observation_space"],
            fast_mock_responses["action_space"],
            fast_mock_responses["reset_state"],
            fast_mock_responses["reset_info"]
        ] + [fast_mock_responses["step_data"]] * 1000  # 1000 steps
        mock_client.step.return_value = {"type": "CommandExecutedSuccessfully", "content": ""}
        
        env = GamaEnv(
            gaml_experiment_path="test.gaml",
            gaml_experiment_name="test_experiment"
        )
        env.reset()
        
        # Measure step performance
        num_steps = 1000
        start_time = time.time()
        
        for _ in range(num_steps):
            env.step(0)
        
        total_time = time.time() - start_time
        steps_per_second = num_steps / total_time
        
        # Should achieve reasonable step rate (> 100 steps/second)
        assert steps_per_second > 100
        
        print(f"Performance: {steps_per_second:.2f} steps/second")
        
        env.close()
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    @patch('gama_gymnasium.core.client.MessageTypes')
    def test_reset_performance(self, mock_message_types, mock_sync_client, fast_mock_responses):
        """Test reset operation performance."""
        mock_message_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
        mock_client = mock_sync_client.return_value
        mock_client.load.return_value = fast_mock_responses["load"]
        mock_client.expression.side_effect = [
            fast_mock_responses["observation_space"],
            fast_mock_responses["action_space"]
        ] + [fast_mock_responses["reset_state"], fast_mock_responses["reset_info"]] * 100  # 100 resets
        mock_client.stop.return_value = {"type": "CommandExecutedSuccessfully", "content": ""}
        
        env = GamaEnv(
            gaml_experiment_path="test.gaml",
            gaml_experiment_name="test_experiment"
        )
        
        # Measure reset performance
        num_resets = 100
        start_time = time.time()
        
        for _ in range(num_resets):
            env.reset()
        
        total_time = time.time() - start_time
        resets_per_second = num_resets / total_time
        
        # Should achieve reasonable reset rate (> 10 resets/second)
        assert resets_per_second > 10
        
        print(f"Reset performance: {resets_per_second:.2f} resets/second")
        
        env.close()
    
    def test_space_conversion_performance(self):
        """Test space conversion performance with various space types."""
        space_definitions = [
            {"type": "Discrete", "n": 10},
            {"type": "Box", "low": [0.0], "high": [1.0], "shape": [1]},
            {"type": "Box", "low": [-1.0] * 100, "high": [1.0] * 100, "shape": [100]},
            {"type": "MultiBinary", "n": 50},
            {"type": "MultiDiscrete", "nvec": [5, 3, 8, 2]},
        ]
        
        # Test conversion speed
        conversions_per_definition = 1000
        
        for definition in space_definitions:
            start_time = time.time()
            
            for _ in range(conversions_per_definition):
                space = map_to_space(definition)
            
            total_time = time.time() - start_time
            conversions_per_second = conversions_per_definition / total_time
            
            # Should convert quickly (> 1000 conversions/second)
            assert conversions_per_second > 1000
            
            print(f"Space {definition['type']}: {conversions_per_second:.0f} conversions/second")
    
    def test_sync_wrapper_overhead(self):
        """Test SyncWrapper performance overhead."""
        
        class MockAsyncEnv:
            """Mock async environment for testing."""
            
            async def reset(self, seed=None):
                await asyncio.sleep(0.001)  # Simulate 1ms delay
                return 0, {}
            
            async def step(self, action):
                await asyncio.sleep(0.001)  # Simulate 1ms delay
                return 1, 1.0, False, False, {}
            
            async def close(self):
                pass
        
        # Test direct async calls
        async def test_direct_async():
            env = MockAsyncEnv()
            
            start_time = time.time()
            
            await env.reset()
            for _ in range(100):
                await env.step(0)
            
            return time.time() - start_time
        
        # Test through SyncWrapper
        def test_sync_wrapper():
            # Mock the sync wrapper behavior
            loop = asyncio.new_event_loop()
            
            def run_sync(coro):
                return loop.run_until_complete(coro)
            
            env = MockAsyncEnv()
            
            start_time = time.time()
            
            run_sync(env.reset())
            for _ in range(100):
                run_sync(env.step(0))
            
            loop.close()
            return time.time() - start_time
        
        # Measure overhead
        direct_time = asyncio.run(test_direct_async())
        sync_time = test_sync_wrapper()
        
        overhead = (sync_time - direct_time) / direct_time * 100
        
        # Overhead should be reasonable (< 50%)
        assert overhead < 50
        
        print(f"SyncWrapper overhead: {overhead:.1f}%")
    
    @patch('gama_gymnasium.core.client.GamaSyncClient')
    @patch('gama_gymnasium.core.client.MessageTypes')
    def test_concurrent_environments(self, mock_message_types, mock_sync_client, fast_mock_responses):
        """Test performance with multiple concurrent environments."""
        mock_message_types.CommandExecutedSuccessfully.value = "CommandExecutedSuccessfully"
        
        def create_mock_client():
            mock_client = Mock()
            mock_client.load.return_value = fast_mock_responses["load"]
            mock_client.expression.side_effect = [
                fast_mock_responses["observation_space"],
                fast_mock_responses["action_space"],
                fast_mock_responses["reset_state"],
                fast_mock_responses["reset_info"]
            ] + [fast_mock_responses["step_data"]] * 100
            mock_client.step.return_value = {"type": "CommandExecutedSuccessfully", "content": ""}
            mock_client.stop.return_value = {"type": "CommandExecutedSuccessfully", "content": ""}
            mock_client.close_connection.return_value = None
            return mock_client
        
        mock_sync_client.side_effect = create_mock_client
        
        def run_environment(env_id):
            """Run a single environment for testing."""
            env = GamaEnv(
                gaml_experiment_path=f"test_{env_id}.gaml",
                gaml_experiment_name="test_experiment"
            )
            
            env.reset()
            for _ in range(100):
                env.step(0)
            
            env.close()
            return env_id
        
        # Test with multiple environments
        num_envs = 4
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_envs) as executor:
            futures = [executor.submit(run_environment, i) for i in range(num_envs)]
            results = [f.result() for f in futures]
        
        total_time = time.time() - start_time
        
        # All environments should complete
        assert len(results) == num_envs
        
        # Should handle concurrent environments efficiently
        assert total_time < 10.0  # Should complete in under 10 seconds
        
        print(f"Concurrent environments: {num_envs} envs completed in {total_time:.2f}s")
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable over many operations."""
        import gc
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate many space conversions (potential memory leak source)
        for _ in range(10000):
            space = map_to_space({"type": "Discrete", "n": 10})
            # Force garbage collection periodically
            if _ % 1000 == 0:
                gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (< 10MB)
        assert memory_increase < 10
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
    
    def test_numpy_array_performance(self):
        """Test performance of numpy array operations."""
        # Test various array sizes
        sizes = [10, 100, 1000, 10000]
        
        for size in sizes:
            # Create test array
            arr = np.random.random(size)
            
            # Test serialization/deserialization performance
            num_operations = 1000
            start_time = time.time()
            
            for _ in range(num_operations):
                # Simulate array processing
                serialized = arr.tolist()
                deserialized = np.array(serialized)
            
            total_time = time.time() - start_time
            operations_per_second = num_operations / total_time
            
            # Should handle arrays efficiently
            assert operations_per_second > 100
            
            print(f"Array size {size}: {operations_per_second:.0f} ops/second")


class TestPerformanceBenchmarks:
    """Benchmark tests for comparison and regression detection."""
    
    @pytest.mark.benchmark
    def test_environment_creation_benchmark(self, benchmark):
        """Benchmark environment creation."""
        with patch('gama_gymnasium.core.client.GamaSyncClient') as mock_sync_client:
            mock_client = mock_sync_client.return_value
            mock_client.load.return_value = {"type": "CommandExecutedSuccessfully", "content": "exp_123"}
            mock_client.expression.side_effect = [
                {"type": "CommandExecutedSuccessfully", "content": {"type": "Discrete", "n": 4}},
                {"type": "CommandExecutedSuccessfully", "content": {"type": "Discrete", "n": 2}}
            ]
            
            def create_env():
                env = GamaEnv("test.gaml", "test_experiment")
                env.close()
                return env
            
            result = benchmark(create_env)
    
    @pytest.mark.benchmark
    def test_space_conversion_benchmark(self, benchmark):
        """Benchmark space conversion."""
        definition = {"type": "Box", "low": [-1.0] * 50, "high": [1.0] * 50, "shape": [50]}
        
        result = benchmark(map_to_space, definition)
