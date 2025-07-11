/**
* Test model with MultiBinary action and observation spaces.
* Used for testing multi-binary space conversion and validation.
*/

model MultiBinaryTestModel

import "../gama/gama gymnasium.gaml"

global {
    // MultiBinary observation space: 8 binary flags
    list<int> current_flags <- [0, 0, 0, 0, 0, 0, 0, 0];
    
    // Environment state
    float current_reward <- 0.0;
    bool is_terminated <- false;
    bool is_truncated <- false;
    map<string, unknown> current_info <- map([]);
    
    // Episode tracking
    int episode_count <- 0;
    int step_count <- 0;
    
    init {
        create MultiBinaryTestAgent number: 1;
        create MultiBinaryGymnasiumManager number: 1;
    }
    
    action reset_environment {
        // Random initial flags
        current_flags <- [];
        loop i from: 0 to: 7 {
            current_flags <- current_flags + rnd(1);
        }
        
        current_reward <- 0.0;
        is_terminated <- false;
        is_truncated <- false;
        current_info <- map(["episode"::episode_count]);
        step_count <- 0;
        episode_count <- episode_count + 1;
    }
    
    action execute_step(list<int> binary_action) {
        step_count <- step_count + 1;
        
        // Apply binary action (toggle flags)
        loop i from: 0 to: min(length(binary_action) - 1, 7) {
            if (binary_action[i] = 1) {
                current_flags[i] <- 1 - current_flags[i];  // Toggle flag
            }
        }
        
        // Calculate reward based on pattern
        int active_flags <- 0;
        loop flag over: current_flags {
            if (flag = 1) {
                active_flags <- active_flags + 1;
            }
        }
        
        // Reward for having exactly 4 flags active (half)
        if (active_flags = 4) {
            current_reward <- 5.0;
        } else {
            current_reward <- float(8 - abs(4 - active_flags));
        }
        
        // Special bonus for specific patterns
        if (current_flags = [1, 0, 1, 0, 1, 0, 1, 0]) {
            current_reward <- current_reward + 10.0;
            is_terminated <- true;
        }
        
        // Terminate after 30 steps
        if (step_count >= 30) {
            is_terminated <- true;
        }
        
        current_info <- map([
            "step"::step_count,
            "active_flags"::active_flags,
            "pattern"::copy(current_flags)
        ]);
    }
}

species MultiBinaryTestAgent {
    list<int> flags <- [0, 0, 0, 0, 0, 0, 0, 0];
    
    action perform_binary_action(list<int> action_values) {
        ask world {
            do execute_step(action_values);
        }
    }
}

species MultiBinaryGymnasiumManager parent: GymnasiumCommunication {
    
    action define_observation_space {
        observation_space <- map([
            "type"::"MultiBinary",
            "n"::8
        ]);
    }
    
    action define_action_space {
        action_space <- map([
            "type"::"MultiBinary",
            "n"::6  // Can toggle 6 of the 8 flags
        ]);
    }
    
    action get_observation {
        observation <- copy(current_flags);
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

experiment multi_binary_test_experiment type: headless {
    
    output {
        display "MultiBinary Environment" {
            chart "Flag Status" {
                data "Active Flags" value: length(current_flags where (each = 1));
                data "Total Flags" value: length(current_flags);
                data "Reward" value: current_reward;
            }
        }
    }
}
