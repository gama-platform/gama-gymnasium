"""
Tests for message handling functionality.
"""

import pytest
from gama_gymnasium.core.message_handler import MessageHandler
from gama_gymnasium.utils.exceptions import MessageValidationError


class TestMessageHandler:
    """Test message handler functionality."""
    
    def test_validate_step_data_valid(self):
        """Test validation of valid step data."""
        valid_data = {
            "State": [1, 2, 3],
            "Reward": 10.5,
            "Terminated": False,
            "Truncated": False,
            "Info": {"episode": 1}
        }
        
        # Should not raise any exception
        MessageHandler.validate_step_data(valid_data)
    
    def test_validate_step_data_missing_fields(self):
        """Test validation with missing required fields."""
        invalid_data = {
            "State": [1, 2, 3],
            "Reward": 10.5
            # Missing Terminated, Truncated, Info
        }
        
        with pytest.raises(MessageValidationError, match="Missing required fields"):
            MessageHandler.validate_step_data(invalid_data)
    
    def test_validate_step_data_invalid_reward_type(self):
        """Test validation with invalid reward type."""
        invalid_data = {
            "State": [1, 2, 3],
            "Reward": "not_a_number",  # Should be numeric
            "Terminated": False,
            "Truncated": False,
            "Info": {}
        }
        
        with pytest.raises(MessageValidationError, match="Reward must be a number"):
            MessageHandler.validate_step_data(invalid_data)
    
    def test_validate_step_data_invalid_terminated_type(self):
        """Test validation with invalid terminated type."""
        invalid_data = {
            "State": [1, 2, 3],
            "Reward": 10.5,
            "Terminated": "true",  # Should be boolean
            "Truncated": False,
            "Info": {}
        }
        
        with pytest.raises(MessageValidationError, match="Terminated must be a boolean"):
            MessageHandler.validate_step_data(invalid_data)
    
    def test_validate_step_data_invalid_info_type(self):
        """Test validation with invalid info type."""
        invalid_data = {
            "State": [1, 2, 3],
            "Reward": 10.5,
            "Terminated": False,
            "Truncated": False,
            "Info": "not_a_dict"  # Should be dictionary
        }
        
        with pytest.raises(MessageValidationError, match="Info must be a dictionary"):
            MessageHandler.validate_step_data(invalid_data)
    
    def test_validate_step_data_not_dict(self):
        """Test validation when data is not a dictionary."""
        with pytest.raises(MessageValidationError, match="Step data must be a dictionary"):
            MessageHandler.validate_step_data("not_a_dict")
    
    def test_validate_space_definition_discrete_valid(self):
        """Test validation of valid discrete space definition."""
        space_def = {
            "type": "Discrete",
            "n": 4
        }
        
        # Should not raise any exception
        MessageHandler.validate_space_definition(space_def)
    
    def test_validate_space_definition_box_valid(self):
        """Test validation of valid box space definition."""
        space_def = {
            "type": "Box",
            "low": [0.0, -1.0],
            "high": [1.0, 1.0],
            "shape": [2]
        }
        
        # Should not raise any exception
        MessageHandler.validate_space_definition(space_def)
    
    def test_validate_space_definition_missing_type(self):
        """Test validation with missing type field."""
        space_def = {"n": 4}
        
        with pytest.raises(MessageValidationError, match="must contain 'type' field"):
            MessageHandler.validate_space_definition(space_def)
    
    def test_validate_space_definition_box_missing_fields(self):
        """Test validation of box space missing required fields."""
        space_def = {
            "type": "Box"
            # Missing low, high, shape
        }
        
        with pytest.raises(MessageValidationError, match="Box space missing required fields"):
            MessageHandler.validate_space_definition(space_def)
    
    def test_validate_space_definition_discrete_missing_n(self):
        """Test validation of discrete space missing n field."""
        space_def = {
            "type": "Discrete"
        }
        
        with pytest.raises(MessageValidationError, match="Discrete space must contain 'n' field"):
            MessageHandler.validate_space_definition(space_def)
    
    def test_validate_space_definition_discrete_invalid_n(self):
        """Test validation of discrete space with invalid n."""
        space_def = {
            "type": "Discrete",
            "n": -1
        }
        
        with pytest.raises(MessageValidationError, match="must be a positive integer"):
            MessageHandler.validate_space_definition(space_def)
    
    def test_validate_space_definition_unsupported_type(self):
        """Test validation with unsupported space type."""
        space_def = {
            "type": "UnsupportedSpace"
        }
        
        with pytest.raises(MessageValidationError, match="Unsupported space type"):
            MessageHandler.validate_space_definition(space_def)
    
    def test_format_action_for_gama_simple(self):
        """Test formatting simple action for GAMA."""
        action = 2
        formatted = MessageHandler.format_action_for_gama(action)
        assert formatted == "2"
    
    def test_format_action_for_gama_list(self):
        """Test formatting list action for GAMA."""
        action = [1, 2, 3]
        formatted = MessageHandler.format_action_for_gama(action)
        assert formatted == "[1, 2, 3]"
    
    def test_format_action_for_gama_dict(self):
        """Test formatting dictionary action for GAMA."""
        action = {"move": "up", "speed": 1.5}
        formatted = MessageHandler.format_action_for_gama(action)
        
        # Should contain the GAMA map format
        assert '"move"::up' in formatted
        assert '"speed"::1.5' in formatted
        assert formatted.startswith('[')
        assert formatted.endswith(']')
    
    def test_parse_gama_response_valid(self):
        """Test parsing valid GAMA response."""
        response = {
            "type": "CommandExecutedSuccessfully",
            "content": {"result": "success"}
        }
        
        content = MessageHandler.parse_gama_response(response)
        assert content == {"result": "success"}
    
    def test_parse_gama_response_invalid_not_dict(self):
        """Test parsing invalid response (not dict)."""
        with pytest.raises(MessageValidationError, match="GAMA response must be a dictionary"):
            MessageHandler.parse_gama_response("not_a_dict")
    
    def test_parse_gama_response_missing_type(self):
        """Test parsing response missing type field."""
        response = {"content": "some content"}
        
        with pytest.raises(MessageValidationError, match="must contain 'type' field"):
            MessageHandler.parse_gama_response(response)
    
    def test_parse_gama_response_missing_content(self):
        """Test parsing response missing content field."""
        response = {"type": "Success"}
        
        with pytest.raises(MessageValidationError, match="must contain 'content' field"):
            MessageHandler.parse_gama_response(response)
    
    def test_create_experiment_parameters_empty(self):
        """Test creating experiment parameters from empty dict."""
        params = {}
        result = MessageHandler.create_experiment_parameters(params)
        assert result == []
    
    def test_create_experiment_parameters_with_values(self):
        """Test creating experiment parameters with values."""
        params = {
            "population_size": 100,
            "mutation_rate": 0.1,
            "enable_logging": True
        }
        
        result = MessageHandler.create_experiment_parameters(params)
        
        assert len(result) == 3
        assert {"name": "population_size", "value": 100} in result
        assert {"name": "mutation_rate", "value": 0.1} in result
        assert {"name": "enable_logging", "value": True} in result
    
    def test_serialize_for_json_numpy_array(self):
        """Test JSON serialization of numpy array."""
        import numpy as np
        
        arr = np.array([1, 2, 3])
        result = MessageHandler.serialize_for_json(arr)
        assert result == [1, 2, 3]
    
    def test_serialize_for_json_custom_object(self):
        """Test JSON serialization of custom object."""
        class CustomObject:
            def __init__(self):
                self.value = 42
                self.name = "test"
        
        obj = CustomObject()
        result = MessageHandler.serialize_for_json(obj)
        assert result == {"value": 42, "name": "test"}
    
    def test_serialize_for_json_simple_types(self):
        """Test JSON serialization of simple types."""
        assert MessageHandler.serialize_for_json(42) == 42
        assert MessageHandler.serialize_for_json("hello") == "hello"
        assert MessageHandler.serialize_for_json([1, 2, 3]) == [1, 2, 3]
        assert MessageHandler.serialize_for_json({"key": "value"}) == {"key": "value"}
