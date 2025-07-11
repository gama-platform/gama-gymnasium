/**
* Test model for GAMA-Gymnasium unit and integration tests.
* This model provides a simple environment for testing the gymnasium integration.
*/

model TestModel

import "../gama/gama gymnasium.gaml"

global {
    // Environment configuration
    int observation_space_n <- 4;
    int action_space_n <- 2;
    
    // Current state of the environment
    int current_state <- 0;
    float current_reward <- 0.0;
    bool is_terminated <- false;
    bool is_truncated <- false;
    map<string, unknown> current_info <- map([]);
    
    // Episode tracking
    int episode_count <- 0;
    int step_count <- 0;
    
    init {
        create TestAgent number: 1;
        create GymnasiumManager number: 1;
    }
    
    // Reset the environment
    action reset_environment {
        current_state <- 0;
        current_reward <- 0.0;
        is_terminated <- false;
        is_truncated <- false;
        current_info <- map(["episode"::episode_count]);
        step_count <- 0;
        episode_count <- episode_count + 1;
    }
    
    // Execute a step in the environment
    action execute_step(int action_value) {
        step_count <- step_count + 1;
        
        // Simple state transition logic for testing
        current_state <- rnd(observation_space_n - 1);
        current_reward <- 1.0;
        
        // Terminate after 10 steps for testing
        if (step_count >= 10) {
            is_terminated <- true;
            current_reward <- 10.0;
        }
        
        current_info <- map(["step"::step_count]);
    }
}

species TestAgent {
    int state <- 0;
    
    action test_action(int action_value) {
        ask world {
            do execute_step(action_value);
        }
    }
}

species GymnasiumManager parent: GymnasiumCommunication {
    
    // Define observation space
    action define_observation_space {
        observation_space <- map([
            "type"::"Discrete",
            "n"::observation_space_n
        ]);
    }
    
    // Define action space  
    action define_action_space {
        action_space <- map([
            "type"::"Discrete",
            "n"::action_space_n
        ]);
    }
    
    // Get current observation
    action get_observation {
        observation <- current_state;
    }
    
    // Get environment info
    action get_info {
        info <- current_info;
    }
    
    // Check if episode is terminated
    bool is_episode_terminated {
        return is_terminated;
    }
    
    // Check if episode is truncated  
    bool is_episode_truncated {
        return is_truncated;
    }
    
    // Get current reward
    float get_reward {
        return current_reward;
    }
    
    // Execute action
    action execute_action(unknown action_data) {
        int action_value <- int(action_data);
        ask world {
            do execute_step(action_value);
        }
    }
    
    // Reset episode
    action reset_episode {
        ask world {
            do reset_environment;
        }
    }
}

experiment test_experiment type: headless {
    
    parameter "Observation space size" var: observation_space_n <- 4;
    parameter "Action space size" var: action_space_n <- 2;
    
    output {
        display "Test Environment" {
            chart "State Evolution" {
                data "Current State" value: current_state;
                data "Reward" value: current_reward;
            }
        }
    }
}
