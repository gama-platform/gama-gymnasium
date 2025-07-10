/**
* Name: basic_env
* Based on the internal empty template. 
* Author: VezP
* Tags: 
*/


model BasicEnv

global {
	// Communication port for the GAMA-Gymnasium connection
	int gama_server_port <- 0;
	
	// Size of the grid environment (4x4 grid)
	int grid_size <- 4;
	
	init {
		// Create a GymAgent to handle the Gymnasium interface
		create GymAgent;
		// Define action space: 4 discrete actions (up, down, right, left)
		GymAgent[0].action_space <- ["type"::"Discrete", "n"::4];
		// Define observation space: 2D coordinates in the grid (x,y positions)
		GymAgent[0].observation_space <- ["type"::"Box", "low"::0, "high"::grid_size, "shape"::[2], "dtype"::"int"];
		
		// Create the learning agent that will move in the environment
		create target_seeking_agent;
		// Randomly select a target cell and mark it as red
		my_grid target_cell <- one_of(my_grid);
		target_cell.color <- #red;
		target_cell.target <- true;
		write "target: " + [target_cell.grid_x, target_cell.grid_y];
	}
	
	// Main simulation loop - executed at each step
	reflex {
		// Ask the learning agent to perform the next action received from the gym
		ask target_seeking_agent {
			do step(int(GymAgent[0].next_action));
		}
		// Update the gym agent's data after the action is completed
		ask GymAgent {
			do update_data;	// must be called at the end
		}
	}
}

// Agent species that handles the Gymnasium interface
species GymAgent skills:[GymnasiumLink];

// Agent that seeks a target cell in the grid environment
species target_seeking_agent {
	GymAgent gym_agent <- GymAgent[0];  // Reference to the gym agent
	my_grid my_cell;  // Current cell position in the grid
	
	init {
		// Initialize agent at a random cell in the grid
		my_cell <- one_of(my_grid);
		location <- my_cell.location;
		// Set initial state as the grid coordinates
		gym_agent.state <- [my_cell.grid_x, my_cell.grid_y];
	}
	
	action step(int action_) {
		// Define action mappings:
		// 0 -> up (move to cell above)
		// 1 -> down (move to cell below)
		// 2 -> right (move to cell on the right)
		// 3 -> left (move to cell on the left)
		
		switch action_ {
			match 0 {
				// Move up if possible, otherwise stay in current cell
				my_cell <- my_cell.cell_up != nil ? my_cell.cell_up : my_cell;
			}
			match 1 {
				// Move down if possible, otherwise stay in current cell
				my_cell <- my_cell.cell_down != nil ? my_cell.cell_down : my_cell;
			}
			match 2 {
				// Move right if possible, otherwise stay in current cell
				my_cell <- my_cell.cell_right != nil ? my_cell.cell_right : my_cell;
			}
			match 3 {
				// Move left if possible, otherwise stay in current cell
				my_cell <- my_cell.cell_left != nil ? my_cell.cell_left : my_cell;
			}
		}

		// Update the agent's visual location
		location <- my_cell.location;
		
		// Update the gym environment state
		gym_agent.state <- [my_cell.grid_x, my_cell.grid_y];
		// Give reward of 1.0 if target reached, 0.0 otherwise
		gym_agent.reward <- my_cell.target ? 1.0 : 0.0;
		// Episode ends when target is reached
		gym_agent.terminated <- my_cell.target;
		// No time limit in this environment
		gym_agent.truncated <- false;
		// No additional info needed
		gym_agent.info <- [];
		
		// Debug output to console
		write "Action: " + action_ + ", State: " + gym_agent.state + ", Reward: " + gym_agent.reward + ", Terminated: " + gym_agent.terminated;
	}
	
	// Visual representation of the agent (blue circle)
	aspect default {
		draw circle(40/grid_size) color: #blue;
	}
}

// Grid cell structure that forms the environment
grid my_grid width: grid_size height: grid_size{
	bool target;  // Flag to mark if this cell is the target
	
	// References to neighboring cells for movement
	my_grid cell_up;
	my_grid cell_down;
	my_grid cell_right;
	my_grid cell_left;
	
	init {
		target <- false;
		// Find neighboring cells in each direction
		cell_up <- neighbors first_with (each.grid_y < grid_y);
		cell_down <- neighbors first_with (each.grid_y > grid_y);
		
		cell_right <- neighbors first_with (each.grid_x > grid_x);
		cell_left <- neighbors first_with (each.grid_x < grid_x);
	}
}

// Experiment for testing the environment without external connections
experiment test_env {
    output {
    	display Render type: 2d {
    		grid my_grid border: #black;
    		species target_seeking_agent aspect: default;
    	}
    }
}

// Experiment for connecting to external Gymnasium environment
experiment gym_env {
	parameter "communication_port" var: gama_server_port;
	parameter "grid size" var: grid_size;
	
	output {
    	display Render type: 2d {
    		grid my_grid border: #black;
    		species target_seeking_agent aspect: default;
    	}
    }
}


















