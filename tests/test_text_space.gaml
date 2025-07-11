/**
* Test model with Text action and observation spaces.
* Used for testing text space conversion and validation.
*/

model TextSpaceTestModel

import "../gama/gama gymnasium.gaml"

global {
    // Text observation space
    string current_message <- "start";
    list<string> message_history <- [];
    
    // Environment state
    float current_reward <- 0.0;
    bool is_terminated <- false;
    bool is_truncated <- false;
    map<string, unknown> current_info <- map([]);
    
    // Episode tracking
    int episode_count <- 0;
    int step_count <- 0;
    
    // Word dictionary for text generation
    list<string> word_dictionary <- ["hello", "world", "test", "gama", "gym", "agent", "action", "reward", "done", "info"];
    
    init {
        create TextTestAgent number: 1;
        create TextGymnasiumManager number: 1;
    }
    
    action reset_environment {
        current_message <- "episode " + episode_count;
        message_history <- [];
        current_reward <- 0.0;
        is_terminated <- false;
        is_truncated <- false;
        current_info <- map(["episode"::episode_count]);
        step_count <- 0;
        episode_count <- episode_count + 1;
    }
    
    action execute_step(string text_action) {
        step_count <- step_count + 1;
        
        // Add action to history
        message_history <- message_history + text_action;
        
        // Process the text action
        current_reward <- 0.0;
        
        // Reward for using words from dictionary
        loop word over: word_dictionary {
            if (contains(text_action, word)) {
                current_reward <- current_reward + 2.0;
            }
        }
        
        // Reward for message length (optimal around 10-20 characters)
        int msg_length <- length(text_action);
        if (msg_length >= 10 and msg_length <= 20) {
            current_reward <- current_reward + 1.0;
        }
        
        // Update current message based on action
        current_message <- text_action + " " + rnd_choice(word_dictionary);
        
        // Truncate message if too long
        if (length(current_message) > 50) {
            current_message <- copy_between(current_message, 0, 50);
        }
        
        // Special termination conditions
        if (contains(text_action, "done") or contains(text_action, "quit")) {
            is_terminated <- true;
            current_reward <- current_reward + 5.0;
        }
        
        // Terminate after 15 steps or if message history gets too long
        if (step_count >= 15 or length(message_history) >= 10) {
            is_terminated <- true;
        }
        
        current_info <- map([
            "step"::step_count,
            "message_length"::length(current_message),
            "history_size"::length(message_history),
            "last_action"::text_action
        ]);
    }
}

species TextTestAgent {
    string message <- "";
    
    action send_text(string text_content) {
        ask world {
            do execute_step(text_content);
        }
    }
}

species TextGymnasiumManager parent: GymnasiumCommunication {
    
    action define_observation_space {
        observation_space <- map([
            "type"::"Text",
            "max_length"::100,
            "charset"::"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?"
        ]);
    }
    
    action define_action_space {
        action_space <- map([
            "type"::"Text",
            "max_length"::50,
            "charset"::"abcdefghijklmnopqrstuvwxyz "
        ]);
    }
    
    action get_observation {
        // Return current message with history context
        string full_observation <- current_message;
        if (length(message_history) > 0) {
            full_observation <- full_observation + " | " + join(message_history, " ");
        }
        
        // Ensure observation fits in max_length
        if (length(full_observation) > 100) {
            full_observation <- copy_between(full_observation, 0, 100);
        }
        
        observation <- full_observation;
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
        string text_action <- string(action_data);
        ask world {
            do execute_step(text_action);
        }
    }
    
    action reset_episode {
        ask world {
            do reset_environment;
        }
    }
}

experiment text_test_experiment type: headless {
    
    output {
        display "Text Environment" {
            chart "Text Metrics" {
                data "Message Length" value: length(current_message);
                data "History Size" value: length(message_history);
                data "Reward" value: current_reward;
            }
        }
    }
}
