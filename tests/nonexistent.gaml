/**
* This file intentionally contains syntax errors for testing error handling.
* It should NOT be loaded successfully by GAMA.
*/

model InvalidModel

// Missing import statement
// import "../gama/gama gymnasium.gaml"

global {
    // Syntax error: missing semicolon
    int invalid_variable <- 0
    
    // Invalid species reference
    init {
        create NonExistentSpecies number: 1;
    }
    
    // Action with syntax error
    action broken_action {
        // Invalid assignment
        this_variable_does_not_exist <- "error";
    }
}

// Species with invalid parent
species BrokenSpecies parent: NonExistentParent {
    
    // Action with invalid syntax
    action invalid_action {
        ask nonexistent_agent {
            do nonexistent_action();
        }
    }
}

// Experiment with errors
experiment broken_experiment type: invalid_type {
    // This should cause errors
}
