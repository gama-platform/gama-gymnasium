/**
* Test model with MultiDiscrete action and observation spaces.
* Used for testing multi-discrete space conversion and validation.
*/

model MultiDiscreteTestModel

import "../gama/gama gymnasium.gaml"

global {
    // MultiDiscrete observation space: [direction (4), intensity (3), mode (2)]
    int current_direction <- 0;  // 0-3: North, East, South, West
    int current_intensity <- 0;  // 0-2: Low, Medium, High
    int current_mode <- 0;       // 0-1: Normal, Special
    
    // Environment state
    float current_reward <- 0.0;
    bool is_terminated <- false;
    bool is_truncated <- false;
    map<string, unknown> current_info <- map([]);
    
    // Episode tracking
    int episode_count <- 0;
    int step_count <- 0;
    
    init {
        create MultiDiscreteTestAgent number: 1;
        create MultiDiscreteGymnasiumManager number: 1;
    }
    
    action reset_environment {
        current_direction <- rnd(3);
        current_intensity <- rnd(2);
        current_mode <- rnd(1);
        current_reward <- 0.0;
        is_terminated <- false;
        is_truncated <- false;
        current_info <- map(["episode"::episode_count]);
        step_count <- 0;
        episode_count <- episode_count + 1;
    }
    
    action execute_step(list<int> multi_action) {
        step_count <- step_count + 1;
        
        // Extract multi-discrete action components
        int action_direction <- multi_action[0];  // 0-3
        int action_intensity <- multi_action[1];  // 0-2
        
        // Update state based on action
        current_direction <- action_direction;
        current_intensity <- action_intensity;
        
        // Calculate reward based on action combination
        current_reward <- 0.0;
        if (action_direction = 1 and action_intensity = 2) {
            // Optimal action combination
            current_reward <- 5.0;
        } else if (action_direction = 0 and action_intensity = 0) {
            // Poor action combination
            current_reward <- -1.0;
        } else {
            current_reward <- 1.0;
        }
        
        // Toggle mode randomly
        current_mode <- rnd(1);
        
        // Terminate after 20 steps
        if (step_count >= 20) {
            is_terminated <- true;
            current_reward <- current_reward + 2.0;
        }
        
        current_info <- map([
            "step"::step_count,
            "direction"::current_direction,
            "intensity"::current_intensity,
            "mode"::current_mode
        ]);
    }
}

species MultiDiscreteTestAgent {
    int direction <- 0;
    int intensity <- 0;
    int mode <- 0;
    
    action perform_multi_action(list<int> action_values) {
        ask world {
            do execute_step(action_values);
        }
    }
}

species MultiDiscreteGymnasiumManager parent: GymnasiumCommunication {
    
    action define_observation_space {
        observation_space <- map([
            "type"::"MultiDiscrete",
            "nvec"::[4, 3, 2]  // [direction, intensity, mode]
        ]);
    }
    
    action define_action_space {
        action_space <- map([
            "type"::"MultiDiscrete",
            "nvec"::[4, 3]  // [direction, intensity]
        ]);
    }
    
    action get_observation {
        observation <- [current_direction, current_intensity, current_mode];
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
        list<int> action_values <- list<int>(action_data);
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

experiment multi_discrete_test_experiment type: headless {
    
    output {
        display "MultiDiscrete Environment" {
            chart "State Components" {
                data "Direction" value: current_direction;
                data "Intensity" value: current_intensity;
                data "Mode" value: current_mode;
                data "Reward" value: current_reward;
            }
        }
    }
}
