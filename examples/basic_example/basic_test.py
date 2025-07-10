"""
Test script for direct GAMA client interaction with basic_env.gaml model.
This script demonstrates low-level communication with GAMA without using the Gymnasium interface.
"""

import json
import asyncio
from pathlib import Path

from gama_client.sync_client import GamaSyncClient
from gama_client.message_types import MessageTypes

async def main():
    # Initialize the GAMA client with connection details
    client = GamaSyncClient("localhost", 1000)

    # Set up experiment parameters
    exp_path = str(Path(__file__).parents[0] / "basic_env.gaml")  # Path to the GAMA model
    exp_name = "gym_env"  # Name of the experiment to run
    exp_parameters = [
        {"type": "float", "name": "seed", "value": 0},  # Random seed for reproducibility
        {"type": "int", "name": "grid_size", "value": 6}  # Size of the grid environment
    ]

    # Step 1: Connect to the GAMA server
    print("Connecting to GAMA server")
    try:
        client.connect()
    except Exception as e:
        print("Error while connecting to the server:", e)
        return

    # Step 2: Load the GAML model and create an experiment
    print("Loading GAML model")
    # Parameters for client.load:
    # - Path to the model file
    # - Experiment name
    # - Console output, reload experiment, auto-run, with outputs
    # - Additional parameters for the experiment
    gama_response = client.load(exp_path, exp_name, False, False, False, True, parameters=exp_parameters)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while loading model:", gama_response)
        return
    
    print("Initialization successful")
    experiment_id = gama_response["content"]  # Store the experiment ID for future commands

    # Step 3: Query the current seed value from GAMA
    print("Getting current seed value")
    gama_response = client.expression(experiment_id, r"seed")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while getting seed:", gama_response)
        return
    # Parse the JSON response to get the actual value
    seed = json.loads(gama_response["content"])
    print("Current seed value:", seed)

    # Step 4: Set a new seed value in GAMA
    print("Setting new seed value")
    expr = r"seed<-0;"  # GAMA expression to set seed to 0
    gama_response = client.expression(experiment_id, expr)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while setting seed:", gama_response)
        return
    print("Seed value set successfully")

    # Step 5: Reload the experiment to apply changes
    print("Reloading the experiment")
    gama_response = client.reload(experiment_id)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while reloading experiment:", gama_response)
        return
    print("Experiment reloaded successfully")

    # Step 6: Verify the seed value after reloading
    print("Verifying seed value after reload")
    gama_response = client.expression(experiment_id, r"seed")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while getting seed after reload:", gama_response)
        return
    seed = json.loads(gama_response["content"])
    print("Current seed value:", seed)

    # Step 7: Check the grid size parameter
    print("Checking grid size parameter")
    gama_response = client.expression(experiment_id, r"grid_size")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while getting grid size:", gama_response)
        return
    grid_size = json.loads(gama_response["content"])
    print("Grid size:", grid_size)

    # Step 8: Get the current state (agent position)
    print("Getting current agent state")
    gama_response = client.expression(experiment_id, r"state")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while getting agent state:", gama_response)
        return
    state = json.loads(gama_response["content"])
    print("Current agent position:", state)

    # Step 9: Send an action to the agent
    print("Sending action to the agent")
    # Action codes: 0=up, 1=down, 2=right, 3=left
    action = 1  # Action 1 = move down
    expr = fr"world.set_action({action})"
    gama_response = client.expression(experiment_id, expr)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while setting action:", gama_response)
        return
    
    # Step 10: Execute one simulation step
    print("Running one simulation step")
    gama_response = client.step(experiment_id, sync=True)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while executing step:", gama_response)
        return
    print("Simulation step executed successfully")
    
    # Step 11: Get the new state after the action
    print("Getting updated agent state")
    gama_response = client.expression(experiment_id, r"state")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("Error while getting updated state:", gama_response)
        return
    new_state = json.loads(gama_response["content"])
    print("New agent position:", new_state)


# Helper function to handle GAMA responses and reduce code repetition
def check_gama_response(response, error_message):
    """
    Check if a GAMA response indicates success.
    
    Args:
        response: The response from a GAMA client command
        error_message: Message to display if the command failed
        
    Returns:
        bool: True if successful, False otherwise
    """
    if response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print(f"{error_message}:", response)
        return False
    return True

if __name__ == "__main__":
    asyncio.run(main())