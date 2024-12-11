/**
* Name: Frozenlake
* Re-implemntation of the Frozen lake example of Gymnasium
* Author: Patrick Taillandier
* Tags: RL
*/

model Frozenlake

global {
	int grid__side_size <- 12;
	
	//maze generation
	bool random_map <- false;
	list<string> data_map <- ["SFFF", "FHFH", "FFFH", "HFFG"];
	float proba_holes <- 0.2;
	
	bool is_slippy <- false;
	int time_limit <- 100;
	
	
	//Q-Learning parameter
	float learning_rate <- 0.1;
	float discount_factor <- 0.5;
	
	
	
	geometry best_path_geom;
	init {
		do generate_maze;
		create elf ;
	}
	
	action generate_maze {
		if random_map or empty(data_map) or (length(data_map) < grid__side_size){
			bool maze_ok <- false;
			loop while: not maze_ok {
				ask cell {
					state <- "F";
				}
				last(cell).state <- "G";
				ask (cell as list) - [first(cell), last(cell)] {
					if flip(proba_holes) {
						state <- "H";
					}
				} 
			
				using topology(cell) {
					path the_path <- path_between((cell where (each.state != "H")), first(cell), last(cell));
					maze_ok <- 	the_path != nil and not empty(the_path.edges);
				}
			}
		} else {
			loop i from: 0 to:grid__side_size -1 {
				string v <- data_map[i];
				loop j from: 0 to:grid__side_size -1 {
					cell[j,i].state <- v at j;
				}
			}
		}
	}
}


species elf {
	
	//define for each state (cell), the expected value of each possible move (neigbors cell)
	map<cell, map<string,float>> q;
	
	int best_path <- #max_int;
		
	cell my_cell <- first(cell);
	
	list<cell> current_path <- [my_cell];
	point location <- my_cell.location;
	string status <- "D";
	
	bool is_arrived <- false;
	int nb_moves <- 0;
	
	
	init {
		loop c over: cell {
			map<string,float> cells;
			if (c.cell_down != nil) {cells["D"] <- 0.0;}
			if (c.cell_up != nil) {cells["U"] <- 0.0;}
			if (c.cell_right != nil) {cells["R"] <- 0.0;}
			if (c.cell_left != nil) {cells["L"] <- 0.0;}
			q[c] <- cells;
		
		}
	}
	
	int reward {
		if (my_cell.state = "G") {
			return 1;
		}
		if (my_cell.state = "H") {
			return -1;
		}
		return 0;
	}
	
	//move behavior
	reflex moving when: not is_arrived {
		map<string,float> actions <- q[my_cell];
		
		//choose as new cell the one that maximise the action value (random selection among the cells with the same value)
		string act <- shuffle(actions.keys) with_max_of (actions[each]);
		
		//value of the move done (to the new cell)
		float val <- actions[act];
		
		
		switch act {
			match "D" {
				do move_down;
			}
			match "U" {
				do move_up;
			}
			match "R" {
				do move_right;
			}
			match "L" {
				do move_left;
			}
		}
		int reward <- reward();
		//update the value for the action done
		actions[act] <- val + learning_rate * (reward + discount_factor *  max(q[my_cell].values) - val);
	
	}
	
	bool move(cell new_cell, string status_name) {
		nb_moves <- nb_moves +1;
		if (new_cell = nil) {
			return false;
		} 
		my_cell <-new_cell;
		status <- status_name;
		//just to display the path
		current_path << my_cell;
		location <- my_cell.location;
		if (my_cell.state in ["H", "G"] or nb_moves >time_limit ) {
			is_arrived <- true;
		}
		
		return true;
	}
	
	bool move_slippy(list<cell> new_cells, list<string> status_names) {
		int index_ <- rnd(2);
		loop  while:(new_cells[index_] = nil ) {
			index_ <- rnd(2);
		}
		return move(new_cells[index_], status_names[index_]);
	}
	
	reflex arrived when: is_arrived {
		if my_cell.state = "G" {
			if (length(current_path) < best_path) {
				best_path <- length(current_path);
				best_path_geom <- line(current_path collect each.location);
			}
			
		//	write "Find the goal in " + length(current_path);
		} else {
		//	write "Failure";
		}
		my_cell <- first(cell);
		location <- my_cell.location; 
		is_arrived <- false;
		current_path <- [my_cell];
		nb_moves <- 0;
	}
	
	bool move_down {
		if is_slippy {
			return move_slippy([my_cell.cell_down,my_cell.cell_right, my_cell.cell_left], ["D","R","L"]);
		} 
		return move(my_cell.cell_down, "D");
	}
	bool move_up {
		if is_slippy {
			return move_slippy([my_cell.cell_up,my_cell.cell_right, my_cell.cell_left], ["U","R","L"]);
		} 
		return move(my_cell.cell_up, "U");
	}
	bool move_right {
		if is_slippy {
			return move_slippy([my_cell.cell_right,my_cell.cell_up, my_cell.cell_down], ["R","U","D"]);
		} 
		return move(my_cell.cell_right, "R");
	}
	bool move_left{
		if is_slippy {
			return move_slippy([my_cell.cell_left,my_cell.cell_up, my_cell.cell_down], ["L","U","D"]);
		} 
		return move(my_cell.cell_left, "L");
	}
	

	aspect default {
		switch status {
			match "D" {
				draw image_file("images/elf_down.png") size: 120 /grid__side_size  at: location;
			}
			match "U" {
				draw image_file("images/elf_up.png") size: 120 /grid__side_size  at: location;

			}
			match "R" {
				draw image_file("images/elf_right.png") size: 120 /grid__side_size  at: location;

			}
			match "L" {
				draw image_file("images/elf_left.png") size: 120 /grid__side_size  at: location;

			}
		}
	}
}

grid cell width: grid__side_size height: grid__side_size neighbors: 4 {
	string state <- "F" among:["S", "G","F", "H"];
	//- "S" for Start tile
    //- "G" for Goal tile
    //- "F" for frozen tile
    //- "H" for a tile with a hole
    
    cell cell_down;
    cell cell_up;
    cell cell_right;
    cell cell_left;
    
    
    init {
    	cell_down <- neighbors first_with (each.grid_y > grid_y);
    	cell_up <- neighbors first_with (each.grid_y < grid_y);
    	
    	cell_right <- neighbors first_with (each.grid_x > grid_x);
    	cell_left <- neighbors first_with (each.grid_x < grid_x);
    	
    }
    
	aspect default {
		switch state {
			match "S" {
				draw shape texture:("images/ice.png") border: #black; 
			}
			match "F" {
				draw shape texture:("images/ice.png") border: #black; 
			}
			match "H" {
				draw shape texture:("images/hole.png") border: #black; 
			}
			match "G" {
				draw shape texture:("images/ice.png") border: #black; 
				draw image_file("images/goal.png") size: 120 /grid__side_size  at: location;
			}
		}
	}
}

experiment Frozenlake type: gui {
	output {
		display map {
			species cell;
			graphics "best path" {
				if (best_path_geom != nil) {
					draw best_path_geom + 0.5 color: #red;
				}
			}
			species elf;
		}
	}
}
