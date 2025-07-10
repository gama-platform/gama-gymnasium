model seedtest

global {
	float seed <- 123.0;
	float r;
    
    reflex {
    	r <- rnd(10.0);
    }
}

experiment test {
	parameter "SEED" var: seed;
}