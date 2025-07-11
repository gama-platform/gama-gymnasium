/**
* Test model that intentionally generates errors for error handling tests.
* Used for testing error propagation and exception handling.
*/

model ErrorTestModel

import "../gama/gama gymnasium.gaml"

global {
    bool should_fail_on_load <- false;
    bool should_fail_on_step <- false;
    bool should_fail_on_reset <- false;
    string error_message <- "Test error";
    
    int current_state <- 0;
    float current_reward <- 0.0;
    bool is_terminated <- false;
    bool is_truncated <- false;
    map<string, unknown> current_info <- map([]);
    
    init {
        create ErrorTestAgent number: 1;
        create ErrorGymnasiumManager number: 1;
        
        // Simulate load error if configured
        if (should_fail_on_load) {
            error "Failed to load experiment: " + error_message;
        }
    }
    
    action reset_environment {
        if (should_fail_on_reset) {
            error "Failed to reset environment: " + error_message;
        }
        
        current_state <- 0;
        current_reward <- 0.0;
        is_terminated <- false;
        is_truncated <- false;
        current_info <- map([]);
    }
    
    action execute_step(int action_value) {
        if (should_fail_on_step) {
            error "Failed to execute step: " + error_message;
        }
        
        current_state <- rnd(3);
        current_reward <- 1.0;
        current_info <- map(["action"::action_value]);
    }
}

species ErrorTestAgent {
    action perform_action(int action_value) {
        ask world {
            do execute_step(action_value);
        }
    }
}

species ErrorGymnasiumManager parent: GymnasiumCommunication {
    
    action define_observation_space {
        observation_space <- map([
            "type"::"Discrete",
            "n"::4
        ]);
    }
    
    action define_action_space {
        action_space <- map([
            "type"::"Discrete",
            "n"::2
        ]);
    }
    
    action get_observation {
        observation <- current_state;
    }
    
    action get_info {
        info <- current_info;
    }
    
    bool is_episode_terminated {
        return is_terminated;
    }
    
    bool is_episode_truncated {
        return is_truncated;
    }
    
    float get_reward {
        return current_reward;
    }
    
    action execute_action(unknown action_data) {
        int action_value <- int(action_data);
        ask world {
            do execute_step(action_value);
        }
    }
    
    action reset_episode {
        ask world {
            do reset_environment;
        }
    }
}

experiment error_test_experiment type: headless {
    
    parameter "Fail on load" var: should_fail_on_load <- false;
    parameter "Fail on step" var: should_fail_on_step <- false;
    parameter "Fail on reset" var: should_fail_on_reset <- false;
    parameter "Error message" var: error_message <- "Test error";
    
}
