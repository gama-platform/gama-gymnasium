import json
import asyncio
from pathlib import Path

from gama_client.sync_client import GamaSyncClient
from gama_client.message_types import MessageTypes

async def main():
    client = GamaSyncClient("localhost", 1000)

    exp_path = str(Path(__file__).parents[0] / "basic_env.gaml")
    exp_name = "gym_env"
    exp_parameters = [{"type": "float", "name": "seed", "value": 0}, {"type": "int", "name": "grid_size", "value": 6}]

    print("connecting to Gama server")
    try:
        client.connect()
    except Exception as e:
        print("error while connecting to the server", e)
        return

    print("loading a gaml model")
    gama_response = client.load(exp_path, exp_name, False, False, False, True, parameters=exp_parameters)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while loading", gama_response)
        return
    print("initialization successful")
    experiment_id = gama_response["content"]

    print("getting seed")
    gama_response = client.expression(experiment_id, r"seed")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting seed", gama_response)
        return
    seed = json.loads(gama_response["content"])
    print("seed:", seed)

    print("setting seed")
    expr = r"seed<-0;"
    gama_response = client.expression(experiment_id, expr)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while setting seed", gama_response)
        return
    print("seed set successfully")

    print("reloading the experiment")
    gama_response = client.reload(experiment_id)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while reloading", gama_response)
        return
    print("reload successful")

    print("getting seed")
    gama_response = client.expression(experiment_id, r"seed")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting seed", gama_response)
        return
    seed = json.loads(gama_response["content"])
    print("seed:", seed)

    print("getting grid_sie")
    gama_response = client.expression(experiment_id, r"grid_size")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting grid_size", gama_response)
        return
    gs = json.loads(gama_response["content"])
    print("grid_size:", gs)

    print("getting state")
    gama_response = client.expression(experiment_id, r"state")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting state", gama_response)
        return
    state = json.loads(gama_response["content"])
    print("state:", state)

    print("setting action")
    action = 1  # Example action, replace with actual action logic
    expr = fr"world.set_action({action})"
    gama_response = client.expression(experiment_id, expr)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while setting action", gama_response)
        return

    print("asking to run a step")
    gama_response = client.step(experiment_id, sync =  True)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while running step", gama_response)
        return
    print("step executed successfully")
    
    print("getting new state after action")
    gama_response = client.expression(experiment_id, r"state")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting new state", gama_response)
        return
    new_state = json.loads(gama_response["content"])
    print("new state:", new_state)


if __name__ == "__main__":
    asyncio.run(main())