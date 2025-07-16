/**
* Name: Spaces test
* Based on the internal empty template. 
* Author: VezP
* Tags: 
*/


model SpacesTest

global {
	int gama_server_port <- 0;
	
	map<string, unknown> box_space1 <- ["type"::"Box", "low"::0.0, "high"::100.0, "shape"::[2], "dtype"::"float"];
	map<string, unknown> box_space2 <- ["type"::"Box", "low"::0, "high"::255, "shape"::[64, 64], "dtype"::"int"];
	map<string, unknown> box_space3 <- ["type"::"Box", "low"::-1.0, "high"::1.0, "shape"::[4, 4, 4], "dtype"::"float"];
	map<string, unknown> box_space4 <- ["type"::"Box", "low"::[-5, -10], "high"::[5, 10], "dtype"::"int"];
	list<map<string, unknown>> box_spaces <- [box_space1, box_space2, box_space3, box_space4];
	
	map<string, unknown> discrete_space1 <- ["type"::"Discrete", "n"::3];
	map<string, unknown> discrete_space2 <- ["type"::"Discrete", "n"::5, "start"::10];
	list<map<string, unknown>> discrete_spaces <- [discrete_space1, discrete_space2];
	
	map<string, unknown> mb_space1 <- ["type"::"MultiBinary", "n"::[4]];
	map<string, unknown> mb_space2 <- ["type"::"MultiBinary", "n"::[2, 2]];
	map<string, unknown> mb_space3 <- ["type"::"MultiBinary", "n"::[4, 8, 8]];
	list<map<string, unknown>> mb_spaces <- [mb_space1, mb_space2, mb_space3];
	
	map<string, unknown> md_space1 <- ["type"::"MultiDiscrete", "nvec"::[3, 5, 2]];
	map<string, unknown> md_space2 <- ["type"::"MultiDiscrete", "nvec"::[10, 5], "start"::[100, 200]];
	map<string, unknown> md_space3 <- ["type"::"MultiDiscrete", "nvec"::[[2, 3], [4, 5]]];
	list<map<string, unknown>> md_spaces <- [md_space1, md_space2, md_space3];
	
	map<string, unknown> text_space <- ["type"::"Text", "min_length"::0, "max_length"::12];
	list<map<string, unknown>> text_spaces <- [text_space];


	map<string, unknown> box_space_t1 <- ["type"::"Box", "low"::-10, "high"::10, "shape"::[2], "dtype"::"int"];
	map<string, unknown> box_space_t2 <- ["type"::"Box", "low"::-100.0, "high"::100.0, "shape"::[10, 10], "dtype"::"float"];
	map<string, unknown> box_space_t3 <- ["type"::"Box", "low"::0, "high"::255, "shape"::[32, 32, 3], "dtype"::"int"];
	map<string, unknown> box_space_t4 <- ["type"::"Box", "low"::[0.0, 10.0, 300.0], "high"::[50.0, 90.0, 600.0], "dtype"::"float"];
	list<map<string, unknown>> box_spaces_t <- [box_space_t1, box_space_t2, box_space_t3, box_space_t4];
	
	map<string, unknown> discrete_space_t1 <- ["type"::"Discrete", "n"::10];
	map<string, unknown> discrete_space_t2 <- ["type"::"Discrete", "n"::5, "start"::3];
	list<map<string, unknown>> discrete_spaces_t <- [discrete_space_t1, discrete_space_t2];
	
	map<string, unknown> mb_space_t1 <- ["type"::"MultiBinary", "n"::[16]];
	map<string, unknown> mb_space_t2 <- ["type"::"MultiBinary", "n"::[3, 3]];
	map<string, unknown> mb_space_t3 <- ["type"::"MultiBinary", "n"::[2, 2, 2]];
	list<map<string, unknown>> mb_spaces_t <- [mb_space_t1, mb_space_t2, mb_space_t3];
	
	map<string, unknown> md_space_t1 <- ["type"::"MultiDiscrete", "nvec"::[10, 2]];
	map<string, unknown> md_space_t2 <- ["type"::"MultiDiscrete", "nvec"::[3, 3], "start"::[1, 5]];
	map<string, unknown> md_space_t3 <- ["type"::"MultiDiscrete", "nvec"::[[10, 5], [20, 8]]];
	list<map<string, unknown>> md_spaces_t <- [md_space_t1, md_space_t2, md_space_t3];
	
	map<string, unknown> text_space_t <- ["type"::"Text", "min_length"::3, "max_length"::10];
	list<map<string, unknown>> text_spaces_t <- [text_space_t];
	
	list box_actions <- [];
	list discrete_actions <- [];
	list mb_actions <- [];
	list md_actions <- [];
	list text_actions <- [];

	action write_actions {
		write "Box Actions:";
		loop act over: box_actions {
			write act;
		}
		write "Discrete Actions:";
		loop act over: discrete_actions {
			write act;
		}
		write "MultiBinary Actions:";
		loop act over: mb_actions {
			write act;
		}
		write "MultiDiscrete Actions:";
		loop act over: md_actions {
			write act;
		}
		write "Text Actions:";
		loop act over: text_actions {
			write act;
		}
	}
}

experiment test {
	
}