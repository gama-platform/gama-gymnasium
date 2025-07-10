import asyncio
from pathlib import Path
import random

from gama_client.sync_client import GamaSyncClient
from gama_client.message_types import MessageTypes

async def main():
    exp_path = str(Path(__file__).parents[0] / "Seed test.gaml")
    exp_name = "test"

    client = GamaSyncClient("localhost", 1001)

    client.connect()

    print("Loading GAML model...")
    gama_response = client.load(exp_path, exp_name)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        raise Exception("error while loading", gama_response)
    experiment_id = gama_response["content"]
    print("Initialization successful, experiment ID:", experiment_id)

    gama_response = client.expression(experiment_id, r"seed")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        raise Exception("error while evaluating expression", gama_response)
    print("Initial value of seed:", gama_response["content"])

    gama_response = client.step(experiment_id, sync=True)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        raise Exception("error while stepping", gama_response)

    gama_response = client.expression(experiment_id, r"r")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        raise Exception("error while evaluating expression", gama_response)
    print("Initial value of r:", gama_response["content"])

    print("\nReloading 5 timens without changing parameters...")
    for i in range(5):

        gama_response = client.reload(experiment_id)
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while reloading", gama_response)
        
        gama_response = client.expression(experiment_id, r"seed")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while evaluating expression after reload", gama_response)
        print("Value of seed:\t", gama_response["content"])

        gama_response = client.step(experiment_id, sync=True)
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while stepping", gama_response)
        
        gama_response = client.expression(experiment_id, r"r")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while evaluating expression after reload", gama_response)
        print("Value of r:\t", gama_response["content"])

    print("\nReloading 5 times with seed set to 0...")
    for i in range(5):

        gama_response = client.reload(experiment_id, parameters=[{"type": "float", "name": "SEED", "value": 0}])
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while reloading", gama_response)
        
        gama_response = client.expression(experiment_id, r"seed")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while evaluating expression after reload", gama_response)
        print("Value of seed:\t", gama_response["content"])

        gama_response = client.step(experiment_id, sync=True)
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while stepping", gama_response)
        
        gama_response = client.expression(experiment_id, r"r")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while evaluating expression after reload", gama_response)
        print("Value of r:\t", gama_response["content"])

    print("\nReloading 5 times with seed set to a random value...")
    for i in range(5):

        gama_response = client.reload(experiment_id)
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while reloading", gama_response)
        
        n = random.randint(0, 999)
        gama_response = client.expression(experiment_id, fr"seed <- {n};")
        
        gama_response = client.expression(experiment_id, r"seed")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while evaluating expression after reload", gama_response)
        print("Value of seed:\t", int(gama_response["content"]))

        gama_response = client.step(experiment_id, sync=True)
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while stepping", gama_response)
        
        gama_response = client.expression(experiment_id, r"r")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            raise Exception("error while evaluating expression after reload", gama_response)
        print("Value of r:\t", gama_response["content"])
    
if __name__ == "__main__":
    asyncio.run(main())