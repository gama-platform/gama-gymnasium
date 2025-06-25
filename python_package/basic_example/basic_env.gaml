/**
* Name: basic_env
* Based on the internal empty template. 
* Author: VezP
* Tags: 
*/


model BasicEnv

global {
	int gama_server_port <- 0;
	int grid_size <- 4;
	
//	map<string, unknown> action_space;
//	map<string, unknown> observation_space;
//
//	list<int> state;
//	float reward;
//	bool terminated;
//	bool truncated;
//	map<string, unknown> inf;
//	
//	int next_action;

	string action_space;
	string observation_space;
	
	init {
		create GymAgent;
		GymAgent[0].action_space <- ["type"::"Discrete", "n"::4];
		GymAgent[0].observation_space <- ["type"::"Box", "low"::0, "high"::grid_size, "shape"::[2], "dtype"::"int"];

		action_space <- to_json(GymAgent[0].action_space);
		observation_space <- to_json(GymAgent[0].observation_space);
		
		create learner_agent;
		my_grid target_cell <- one_of(my_grid);
		target_cell.color <- #red;
		target_cell.target <- true;
		write "target: " + [target_cell.grid_x, target_cell.grid_y];
	}
	
	reflex {
		ask learner_agent {
			do step(int(GymAgent[0].next_action));
		}
	}
}

species GymAgent {
	map<string, unknown> action_space;
	map<string, unknown> observation_space;
	
	unknown state;
	float reward;
	bool terminated;
	bool truncated;
	map<string, unknown> info;
	
	unknown next_action;
}

species learner_agent {
	GymAgent gym_agent <- GymAgent[0];
	my_grid my_cell;
	
	init {
		my_cell <- one_of(my_grid);
		location <- my_cell.location;
		gym_agent.state <- [my_cell.grid_x, my_cell.grid_y];
	}
	
	action step(int action_) {
		// 0 -> up
		// 1 -> down
		// 2 -> right
		// 3 -> left
		
		switch action_ {
			match 0 {
				my_cell <- my_cell.cell_up != nil ? my_cell.cell_up : my_cell;
			}
			match 1 {
				my_cell <- my_cell.cell_down != nil ? my_cell.cell_down : my_cell;
			}
			match 2 {
				my_cell <- my_cell.cell_right != nil ? my_cell.cell_right : my_cell;
			}
			match 3 {
				my_cell <- my_cell.cell_left != nil ? my_cell.cell_left : my_cell;
			}
		}

		location <- my_cell.location;
		
		gym_agent.state <- [my_cell.grid_x, my_cell.grid_y];
		gym_agent.reward <- my_cell.target ? 1.0 : 0.0;
		gym_agent.terminated <- my_cell.target;
		gym_agent.truncated <- false;
		gym_agent.info <- [];
		
		write "Action: " + action_ + ", State: " + gym_agent.state + ", Reward: " + gym_agent.reward + ", Terminated: " + gym_agent.terminated;
	}
	
	aspect default {
		draw circle(40/grid_size) color: #blue;
	}
}

grid my_grid width: grid_size height: grid_size{
	bool target;
	
	my_grid cell_up;
	my_grid cell_down;
	my_grid cell_right;
	my_grid cell_left;
	
	init {
		target <- false;
    	cell_up <- neighbors first_with (each.grid_y < grid_y);
    	cell_down <- neighbors first_with (each.grid_y > grid_y);
    	
    	cell_right <- neighbors first_with (each.grid_x > grid_x);
    	cell_left <- neighbors first_with (each.grid_x < grid_x);
	}
}

experiment test_env {
    output {
    	display Render type: 2d {
    		grid my_grid border: #black;
    		species learner_agent aspect: default;
    	}
    }
}

experiment gym_env {
	parameter "communication_port" var: gama_server_port;
	parameter "gride size" var: grid_size;
	
	output {
    	display Render type: 2d {
    		grid my_grid border: #black;
    		species learner_agent aspect: default;
    	}
    }
}


















