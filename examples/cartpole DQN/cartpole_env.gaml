/**
* Name: cartpole_env
* Based on the internal empty template. 
* Author: VezP
* Tags: 
*/


model CartpoleEnv

global{
	int gama_server_port <- 0;
	
	int next_action;
	
	bool _sutton_barto_reward <- false;
	
//	int screen_width <- 600;
//	int screen_height <- 400;
	int screen_width <- 100;
	int screen_height <- 100;
	
	float gravity <- 9.8;
	float masscart <- 1.0;
	float masspole <- 0.1;
	float total_mass <- masspole + masscart;
	float length <- 0.5; // actually half the pole's lenght
	float polemass_length <- masspole * length;
	float force_mag <- 10.0;
	float tau <- 0.02;
	string kinematics_integrator <- "euler";
	
	// Angle at wich to fail the episode
	float theta_threshold_radians <- 12 * 2 * #pi / 360;
	float x_threshold <- 2.4;
	
	int steps_beyond_terminated;
	
	list<float> step_times <- [];
	
	init {
		create GymAgent;
		GymAgent[0].action_space <- ["type"::"Discrete", "n"::2];
		GymAgent[0].observation_space <- [
			"type"::"Box", 
			"low"::[-(x_threshold * 2), "-Infinity", -(theta_threshold_radians * 2), "-Infinity"], 
			"high"::[x_threshold * 2, "Infinity", theta_threshold_radians * 2, "Infinity"],
			"dtype"::"float"
			];
		
		create Cartpole;
		steps_beyond_terminated <- -1;
	}

	reflex {
//		float start_step <- gama.machine_time;
		ask Cartpole {
			do step_(int(gym_agent.next_action));
		}
//		float end_step <- gama.machine_time;
//		write "Step time: " + (end_step - start_step);
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
	
	map data;
	
	action update {
		data <- ["State"::state, "Reward"::reward, "Terminated"::terminated, "Truncated"::truncated, "Info"::info];
	}
	
	unknown next_action;
}

species Cartpole {
	
	GymAgent gym_agent <- GymAgent[0];
	
	init {
		gym_agent.state <- list_with(4, rnd(-0.05, 0.05));
		gym_agent.info <- [];
	}
	
	action step_(int action_) {
		
		float x <- float(list(gym_agent.state)[0]);
		float x_dot <- float(list(gym_agent.state)[1]);
		float theta <- float(list(gym_agent.state)[2]);
		float theta_dot <- float(list(gym_agent.state)[3]);
		
		float force <- int(action_) = 1 ? force_mag : -force_mag;
		float costheta <- cos_rad(theta);
		float sintheta <- sin_rad(theta);
		
		float temp <- (force + polemass_length * (theta_dot ^ 2) * sintheta) / total_mass;
		float thetaacc <- (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * (costheta ^ 2) / total_mass));
		float xacc <- temp - polemass_length * thetaacc * costheta / total_mass;
		
		if kinematics_integrator = "euler" {
			x <- x + tau * x_dot;
			x_dot <- x_dot + tau * xacc;
			theta <- theta + tau * theta_dot;
			theta_dot <- theta_dot +tau * thetaacc;
		}
		else {
			x_dot <- x_dot + tau * xacc;
			x <- x + tau * x_dot;
			theta_dot <- theta_dot + tau * thetaacc;
			theta <- theta + tau * theta_dot;
		}
		
		gym_agent.state <- [x, x_dot, theta, theta_dot];
		
		gym_agent.terminated <- bool(
			x < -x_threshold
			or x > x_threshold
			or theta < -theta_threshold_radians
			or theta > theta_threshold_radians
		);
		
		if !gym_agent.terminated {
			gym_agent.reward <- _sutton_barto_reward ? 0.0 : 1.0; 
		}
		else if steps_beyond_terminated = -1 {
			// Poll just fell!
			steps_beyond_terminated <- 0;
			
			gym_agent.reward <- _sutton_barto_reward ? -1.0 : 1.0;
		}
		else {
			if steps_beyond_terminated = 0{
				write "You are calling 'step()' even though this environment has already returned terminated = True.";
				write "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior.";
			}
			steps_beyond_terminated <- steps_beyond_terminated + 1;
			
			gym_agent.reward <- _sutton_barto_reward ? -1.0 : 0.0;
		}
		
		gym_agent.truncated <- false;
		gym_agent.info <- [];
		ask gym_agent {do update;}
	}
	
	aspect default{
		
		float world_width <- x_threshold * 2;
		float scale <- screen_width / world_width;
		float polewidth <- 0.1 * scale;
		float polelen <- (2 * length)* scale;
		float cartwidth <- 0.5 * scale;
		float cartheight <- 0.3 * scale;
		
		list<float> x <- gym_agent.state;
		
		float l <- -cartwidth / 2;
		float r <- cartwidth / 2;
		float t <- cartheight / 2;
		float b <- -cartheight / 2;
		float axleoffset <- cartheight / 4.0;
		float cartx <- x[0] * scale + screen_width / 2.0; // MIDDLE OF CART
		float carty <- screen_height / 4; // TOP OF CART
		
		list<point> cart_coords <- [{l, b}, {l, t}, {r, t}, {r, b}];
		cart_coords <- cart_coords collect ({each.x + cartx, each.y + carty});
		
		geometry cart_poly <- polygon(cart_coords);
		draw cart_poly color: #black;
		
		l <- -polewidth / 2;
		r <- polewidth / 2;
		t <- polelen - polewidth / 2;
		b <- -polewidth / 2;
		
		list<point> pole_coords <- [];
		loop coord over: [{l, b}, {l, t}, {r, t}, {r, b}]{
			coord <- coord rotated_by ((-x[2]) * #to_deg :: {0, 0, 1});
			coord <- {coord.x + cartx, coord.y + carty + axleoffset};
			pole_coords << coord;
		}
		
		geometry pole_poly <- polygon(pole_coords);
		draw pole_poly color: #burlywood;
		
		geometry joint_circle <- circle(int(polewidth / 2), {cartx, carty + axleoffset});
		draw joint_circle color: #lightsteelblue;
		
		draw line({0, carty},{screen_width, carty}) color: #black;
	}
}

experiment gym_env {
	parameter "communication_port" var: gama_server_port;
	
	float world_width <- x_threshold * 2;
	float scale <- 100 / world_width;
	float polewidth <- 10.0 / scale;
	float polelen <- scale * (2 * length);
	float cartwidth <- 50.0 / scale;
	float cartheight <- 30.0 / scale;

	output {
		display Render type: 2d axes: true{
			// camera 'default' location: {50.0464,40.2342,139.659} target: {50.0,50.0,0.0};
			species Cartpole;
			graphics screen_border {
				draw square(99.99) wireframe: true;
			}
		}	
	}
}

















