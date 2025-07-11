/**
* Test model with Box (continuous) action and observation spaces.
* Used for testing continuous space conversion and validation.
*/

model BoxSpaceTestModel

import "../gama/gama gymnasium.gaml"

global {
    // Box observation space (2D position)
    float current_x <- 0.0;
    float current_y <- 0.0;
    
    // Rewards and episode state
    float current_reward <- 0.0;
    bool is_terminated <- false;
    bool is_truncated <- false;
    map<string, unknown> current_info <- map([]);
    
    // Episode tracking
    int episode_count <- 0;
    int step_count <- 0;
    
    init {
        create BoxTestAgent number: 1;
        create BoxGymnasiumManager number: 1;
    }
    
    action reset_environment {
        current_x <- 0.0;
        current_y <- 0.0;
        current_reward <- 0.0;
        is_terminated <- false;
        is_truncated <- false;
        current_info <- map(["episode"::episode_count]);
        step_count <- 0;
        episode_count <- episode_count + 1;
    }
    
    action execute_step(list<float> continuous_action) {
        step_count <- step_count + 1;
        
        // Update position based on continuous action
        float dx <- continuous_action[0];
        float dy <- continuous_action[1];
        
        current_x <- current_x + dx * 0.1;
        current_y <- current_y + dy * 0.1;
        
        // Keep in bounds [-1, 1]
        current_x <- max(-1.0, min(1.0, current_x));
        current_y <- max(-1.0, min(1.0, current_y));
        
        // Reward based on distance to origin
        float distance <- sqrt(current_x^2 + current_y^2);
        current_reward <- 1.0 - distance;
        
        // Terminate if reached goal or max steps
        if (distance < 0.1 or step_count >= 50) {
            is_terminated <- true;
            if (distance < 0.1) {
                current_reward <- 10.0;
            }
        }
        
        current_info <- map(["step"::step_count, "distance"::distance]);
    }
}

species BoxTestAgent {
    float x <- 0.0;
    float y <- 0.0;
    
    action move(list<float> action_values) {
        ask world {
            do execute_step(action_values);
        }
    }
}

species BoxGymnasiumManager parent: GymnasiumCommunication {
    
    action define_observation_space {
        observation_space <- map([
            "type"::"Box",
            "low"::[-1.0, -1.0],
            "high"::[1.0, 1.0],
            "shape"::[2]
        ]);
    }
    
    action define_action_space {
        action_space <- map([
            "type"::"Box",
            "low"::[-1.0, -1.0],
            "high"::[1.0, 1.0],
            "shape"::[2]
        ]);
    }
    
    action get_observation {
        observation <- [current_x, current_y];
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
        list<float> action_values <- list<float>(action_data);
        ask world {
            do execute_step(action_values);
        }
    }
    
    action reset_episode {
        ask world {
            do reset_environment;
        }
    }
}

experiment box_test_experiment type: headless {
    
    output {
        display "Continuous Environment" {
            chart "Position" {
                data "X Position" value: current_x;
                data "Y Position" value: current_y;
                data "Reward" value: current_reward;
            }
        }
    }
}
