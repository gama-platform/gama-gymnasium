# GAMA Client Test Script

This file (`basic_test.py`) demonstrates how to directly interact with a GAMA simulation using the `gama_client` package without using the Gymnasium interface. It provides a low-level view of the communication between Python and GAMA.

## Purpose

The script shows how to:

1. Connect to a GAMA server
2. Load a model and create an experiment
3. Query and modify simulation parameters
4. Execute simulation steps
5. Get simulation state information

This serves as a technical demonstration of the underlying API that powers the Gymnasium integration.

## Key Concepts

### GAMA Synchronous Client

The script uses `GamaSyncClient` to communicate with GAMA. This client handles the socket communication and provides a synchronous API to interact with GAMA.

```python
from gama_client.sync_client import GamaSyncClient
client = GamaSyncClient("localhost", 1000)
```

### Communication Pattern

The typical pattern for GAMA communication is:

1. Send a command via the client
2. Get a response
3. Check if the command was successful
4. Process the response content (often parsing JSON)

### Experiment Parameters

The script shows how to pass parameters to a GAMA experiment:

```python
exp_parameters = [
    {"type": "float", "name": "seed", "value": 0},
    {"type": "int", "name": "grid_size", "value": 6}
]
```

### GAMA Expressions

To query or modify simulation variables, the script uses GAMA expressions:

```python
# Query an expression
client.expression(experiment_id, r"seed")

# Set a value
client.expression(experiment_id, r"seed<-0;")
```

### Step Execution

The script demonstrates how to execute a single simulation step:

```python
client.step(experiment_id, sync=True)
```

## Running the Script

To run this script:

1. Make sure GAMA is running and configured to accept socket connections
2. Ensure you have the `gama_client` package installed
3. Run the script: `python basic_test.py`

## Difference from Gymnasium Interface

This script shows the raw communication with GAMA, while the Gymnasium interface (`basic_gymnasium.py`) provides a standardized API following the OpenAI Gym pattern, hiding these implementation details.

The main differences are:

- Direct connection management vs. abstracted connection
- Manual action passing vs. standardized `step()` method
- Manual state querying vs. automatic state updates
- Raw JSON handling vs. structured observation and reward objects
