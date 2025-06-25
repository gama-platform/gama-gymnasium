import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Text, Tuple, Dict, Sequence, Graph, OneOf

import asyncio
from pathlib import Path

from gama_client.sync_client import GamaSyncClient
from gama_client.message_types import MessageTypes


box_space1 = Box(low=-10, high=10, shape=(2,), dtype=np.int32)
box_space2 = Box(low=-100.0, high=100.0, shape=(10, 10), dtype=np.float32)
box_space3 = Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
box_space4 = Box(low=np.array([0.0, 10.0, 300.0]), high=np.array([50.0, 90.0, 600.0]), dtype=np.float32)
box_spaces = [box_space1, box_space2, box_space3, box_space4]

discrete_space1 = Discrete(10)
discrete_space2 = Discrete(5, start=3)
discrete_spaces = [discrete_space1, discrete_space2]

mb_space1 = MultiBinary(16)
mb_space2 = MultiBinary([3, 3])
mb_space3 = MultiBinary([2, 2, 2])
mb_spaces = [mb_space1, mb_space2, mb_space3]

md_space1 = MultiDiscrete([10, 2])
md_space2 = MultiDiscrete(nvec=[3, 3], start=[1, 5])
md_space3 = MultiDiscrete([[10, 5], [20, 8]])
md_spaces = [md_space1, md_space2, md_space3]

text_space = Text(min_length=3, max_length=10)
text_spaces = [text_space]


box_space_t1 = Box(low=0.0, high=100.0, shape=(2,), dtype=np.float32)
box_space_t2 = Box(low=0, high=255, shape=(64, 64), dtype=np.int32)
box_space_t3 = Box(low=-1.0, high=1.0, shape=(4, 4, 4), dtype=np.float32)
box_space_t4 = Box(low=np.array([-5, -10]), high=np.array([5, 10]), dtype=np.int32)
box_spaces_t = [box_space_t1, box_space_t2, box_space_t3, box_space_t4]

discrete_space_t1 = Discrete(3)
discrete_space_t2 = Discrete(5, start=10)
discrete_spaces_t = [discrete_space_t1, discrete_space_t2]

mb_space_t1 = MultiBinary(4)
mb_space_t2 = MultiBinary([2, 2])
mb_space_t3 = MultiBinary([4, 8, 8])
mb_spaces_t = [mb_space_t1, mb_space_t2, mb_space_t3]

md_space_t1 = MultiDiscrete([3, 5, 2])
md_space_t2 = MultiDiscrete(nvec=[10, 5], start=[100, 200])
md_space_t3 = MultiDiscrete([[2, 3], [4, 5]])
md_spaces_t = [md_space_t1, md_space_t2, md_space_t3]

text_space_t = Text(min_length=0, max_length=12)
text_spaces_t = [text_space_t]


box_actions = []
discrete_actions = []
mb_actions = []
md_actions = []
text_actions = []

for box in box_spaces:
    box_actions.append(box.sample())
for discrete in discrete_spaces:
    discrete_actions.append(discrete.sample())
for mb in mb_spaces:
    mb_actions.append(mb.sample())
for md in md_spaces:
    md_actions.append(md.sample())
for text in text_spaces:
    text_actions.append(text.sample())

async def gama_server_message_handler(message: dict):
    print("GAMA message received:", message["content"]["message"])

def server_connection(client):
    exp_path = str(Path(__file__).parents[0] / "Spaces test.gaml")
    exp_name = "test"

    try:
        client.connect()
    except Exception as e:
        print("error while connecting to the server", e)
        return

    gama_response = client.load(exp_path, exp_name, True, False, False, True)
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while loading", gama_response)
        return
    experiment_id = gama_response["content"]

    return experiment_id

def get_msgs(client, experiment_id, msgs):
    returned_msgs = []
    for m in msgs:
        gama_response = client.expression(experiment_id, m)
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            print(f"error while getting message '{m}'", gama_response)
            return
        print(f"message '{m}' received successfully")
        returned_msgs.append(json.loads(gama_response["content"]))
    return returned_msgs

def map_to_space(map):
    if "type" in map:
        if map["type"] == "Discrete":
            return map_to_discrete(map)
        elif map["type"] == "Box":
            return map_to_box(map)
        elif map["type"] == "MultiBinary":
            return map_to_multi_binary(map)
        elif map["type"] == "MultiDiscrete":
            return map_to_multi_discrete(map)
        elif map["type"] == "Text":
            return map_to_text(map)
        # elif map["type"] == "Tuple":
        #     return map_to_tuple(map)
        # elif map["type"] == "Dict":
        #     return map_to_dict(map)
        # elif map["type"] == "Sequence":
        #     return map_to_sequence(map)
        # elif map["type"] == "Graph":
        #     return map_to_graph(map)
        # elif map["type"] == "OneOf":
        #     return map_to_one_of(map)
        else:
            print("Unknown type in the map, cannot map to space.")
            return None
    else:
        print("No type specified in the map, cannot map to space.")
        return None

def map_to_box(box):
    if "low" in box:
        if isinstance(box["low"], list):
            low = np.array(box["low"])
        else:
            low = box["low"]
    else:
        low = -np.inf
    
    if "high" in box:
        if isinstance(box["high"], list):
            high = np.array(box["high"])
        else:
            high = box["high"]
    else:
        high = np.inf

    if "shape" in box:
        shape = box["shape"]
    else:
        shape = None

    if "dtype" in box:
        if box["dtype"] == "int":
            dtype = np.int32
        elif box["dtype"] == "float":
            dtype = np.float32
        else:
            print("Unknown dtype in the box, defaulting to float32.")
            dtype = np.float32
    else:
        dtype = np.float32

    return Box(low=low, high=high, shape=shape, dtype=dtype)

def map_to_discrete(discrete):
    n = discrete["n"]
    if "start" in discrete:
        start = discrete["start"]
        return Discrete(n, start=start)
    else:
        return Discrete(n)

def map_to_multi_binary(mb):
    n = mb["n"]
    if len(n) == 1:
        return MultiBinary(n[0])
    else:
        return MultiBinary(n)

def map_to_multi_discrete(md):
    nvec = md["nvec"]
    if "start" in md:
        start = md["start"]
        return MultiDiscrete(nvec, start=start)
    else:
        return MultiDiscrete(nvec)

def map_to_text(text):
    if "min_length" in text:
        min = text["min_length"]
    else:
        min = 0

    if "max_length" in text:
        max = text["max_length"]
    else:
        max = 1000
        
    return Text(min_length=min, max_length=max)

def test_receive_spaces(client, experiment_id):

    print("getting box_map")
    box_map = []
    gama_response = client.expression(experiment_id, "box_space1")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting box_space1", gama_response)
        return
    box_map1 = json.loads(gama_response["content"])
    print("box_map1:", box_map1)
    box_map.append(box_map1)

    gama_response = client.expression(experiment_id, "box_space2")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting box_space2", gama_response)
        return
    box_map2 = json.loads(gama_response["content"])
    print("box_map2:", box_map2)
    box_map.append(box_map2)

    gama_response = client.expression(experiment_id, "box_space3")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting box_space3", gama_response)
        return
    box_map3 = json.loads(gama_response["content"])
    print("box_map3:", box_map3)
    box_map.append(box_map3)

    gama_response = client.expression(experiment_id, "box_space4")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting box_space4", gama_response)
        return
    box_map4 = json.loads(gama_response["content"])
    print("box_map4:", box_map4)
    box_map.append(box_map4)


    print("\ngetting discrete_map")
    discrete_map = []
    gama_response = client.expression(experiment_id, "discrete_space1")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting discrete_space1", gama_response)
        return
    discrete_map1 = json.loads(gama_response["content"])
    print("discrete_map1:", discrete_map1)
    discrete_map.append(discrete_map1)

    gama_response = client.expression(experiment_id, "discrete_space2")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting discrete_space2", gama_response)
        return
    discrete_map2 = json.loads(gama_response["content"])
    print("discrete_map2:", discrete_map2)
    discrete_map.append(discrete_map2)


    print("\ngetting multi_binary_map")
    mb_map = []
    gama_response = client.expression(experiment_id, "mb_space1")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting multi_binary_space1", gama_response)
        return
    mb_map1 = json.loads(gama_response["content"])
    print("multi_binary_map1:", mb_map1)
    mb_map.append(mb_map1)

    gama_response = client.expression(experiment_id, "mb_space2")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting multi_binary_space2", gama_response)
        return
    mb_map2 = json.loads(gama_response["content"])
    print("multi_binary_map2:", mb_map2)
    mb_map.append(mb_map2)

    gama_response = client.expression(experiment_id, "mb_space3")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting multi_binary_space3", gama_response)
        return
    mb_map3 = json.loads(gama_response["content"])
    print("multi_binary_map3:", mb_map3)
    mb_map.append(mb_map3)


    print("\ngetting multi_discrete_map")
    md_map = []
    gama_response = client.expression(experiment_id, "md_space1")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting multi_discrete_space1", gama_response)
        return
    md_map1 = json.loads(gama_response["content"])
    print("multi_discrete_map1:", md_map1)
    md_map.append(md_map1)

    gama_response = client.expression(experiment_id, "md_space2")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting multi_discrete_space2", gama_response)
        return
    md_map2 = json.loads(gama_response["content"])
    print("multi_discrete_map2:", md_map2)
    md_map.append(md_map2)

    gama_response = client.expression(experiment_id, "md_space3")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting multi_discrete_space3", gama_response)
        return
    md_map3 = json.loads(gama_response["content"])
    print("multi_discrete_map3:", md_map3)
    md_map.append(md_map3)


    print("\ngetting text_space")
    text_map = []
    gama_response = client.expression(experiment_id, "text_space")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting text_space", gama_response)
        return
    text_map1 = json.loads(gama_response["content"])
    print("text_map:", text_map1)
    text_map.append(text_map1)

    
    print("\nChecking received spaces against expected spaces...")
    print("Box spaces:")
    for i, box in enumerate(box_map):
        box_space = map_to_space(box)
        if box_space != box_spaces_t[i]:
            print(f"Box space {i} does not match expected space.")
            print(f"Expected: {box_spaces_t[i]}, Received: {box_space}")
        else:
            print(f"Box space {i} matches expected space.")
            print(f"box_space{i}: {box_space}")
    
    print("\nDiscrete spaces:")
    for i, discrete in enumerate(discrete_map):
        discrete_space = map_to_space(discrete)
        if discrete_space != discrete_spaces_t[i]:
            print(f"Discrete space {i} does not match expected space.")
            print(f"Expected: {discrete_spaces_t[i]}, Received: {discrete_space}")
        else:
            print(f"Discrete space {i} matches expected space.")
            print(f"discrete_space{i}: {discrete_space}")

    print("\nMulti-binary spaces:")
    for i, mb in enumerate(mb_map):
        mb_space = map_to_space(mb)
        if mb_space != mb_spaces_t[i]:
            print(f"Multi-binary space {i} does not match expected space.")
            print(f"Expected: {mb_spaces_t[i]}, Received: {mb_space}")
        else:
            print(f"Multi-binary space {i} matches expected space.")
            print(f"mb_space{i}: {mb_space}")

    print("\nMulti-discrete spaces:")
    for i, md in enumerate(md_map):
        md_space = map_to_space(md)
        if md_space != md_spaces_t[i]:
            print(f"Multi-discrete space {i} does not match expected space.")
            print(f"Expected: {md_spaces_t[i]}, Received: {md_space}")
        else:
            print(f"Multi-discrete space {i} matches expected space.")
            print(f"md_space{i}: {md_space}")

    print("\nText spaces:")
    for i, text in enumerate(text_map):
        text_space = map_to_space(text)
        if text_space != text_spaces_t[i]:
            print(f"Text space {i} does not match expected space.")
            print(f"Expected: {text_spaces_t[i]}, Received: {text_space}")
        else:
            print(f"Text space {i} matches expected space.")
            print(f"text_space{i}: {text_space}")


def test_send_actions(client, experiment_id):
    for i, box in enumerate(box_actions):
        box = box_spaces[i].to_jsonable([box])[0]
        gama_response = client.expression(experiment_id, f"add item: {json.dumps(box)} to: box_actions;")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            print(f"Error while sending box action {box}: {gama_response}")
            return
        print(f"Box action {box} from space {box_spaces[i]} sent successfully.")

    for i, discrete in enumerate(discrete_actions):
        discrete = discrete_spaces[i].to_jsonable([discrete])[0]
        gama_response = client.expression(experiment_id, f"add item: {json.dumps(discrete)} to: discrete_actions;")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            print(f"Error while sending discrete action {discrete}: {gama_response}")
            return
        print(f"Discrete action {discrete} from space {discrete_spaces[i]} sent successfully.")

    for i, mb in enumerate(mb_actions):
        mb = mb_spaces[i].to_jsonable([mb])[0]
        gama_response = client.expression(experiment_id, f"add item: {json.dumps(mb)} to: mb_actions;")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            print(f"Error while sending multi-binary action {mb}: {gama_response}")
            return
        print(f"Multi-binary action {mb} from space {mb_spaces[i]} sent successfully.")

    for i, md in enumerate(md_actions):
        md = md_spaces[i].to_jsonable([md])[0]
        gama_response = client.expression(experiment_id, f"add item: {json.dumps(md)} to: md_actions;")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            print(f"Error while sending multi-discrete action {md}: {gama_response}")
            return
        print(f"Multi-discrete action {md} from space {md_spaces[i]} sent successfully.")

    for i, text in enumerate(text_actions):
        text = text_spaces[i].to_jsonable([text])[0]
        gama_response = client.expression(experiment_id, f"add item: {json.dumps(text)} to: text_actions;")
        if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
            print(f"Error while sending text action {text}: {gama_response}")
            return
        print(f"Text action {text} from space {text_spaces[i]} sent successfully.\n")

    gama_response = client.expression(experiment_id, "world.write_actions()")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print(f"Error while sending text action {text}: {gama_response}")
        return
    
def test_receive_states(client, experiment_id):
    box_states = []
    discrete_states = []
    mb_states = []
    md_states = []
    text_states = []

    gama_response = client.expression(experiment_id, "box_actions")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting box_actions", gama_response)
        return
    box_states = json.loads(gama_response["content"])

    gama_response = client.expression(experiment_id, "discrete_actions")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting discrete_actions", gama_response)
        return
    discrete_states = json.loads(gama_response["content"])

    gama_response = client.expression(experiment_id, "mb_actions")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting mb_actions", gama_response)
        return
    mb_states = json.loads(gama_response["content"])

    gama_response = client.expression(experiment_id, "md_actions")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting md_actions", gama_response)
        return
    md_states = json.loads(gama_response["content"])

    gama_response = client.expression(experiment_id, "text_actions")
    if gama_response["type"] != MessageTypes.CommandExecutedSuccessfully.value:
        print("error while getting text_actions", gama_response)
        return
    text_states = json.loads(gama_response["content"])

    print("\nBox states:")
    for i, box in enumerate(box_states):
        box = box_spaces[i].from_jsonable([box])[0]
        if not box_spaces[i].contains(box):
            print(f"Box state {i} does not match expected space.")
            print(f"Expected space: {box_spaces[i]}")
            print(f"Expected value: {box_actions[i]}, Received: {box}")
        else:
            print(f"Box state {i} matches expected state.")

    print("\nDiscrete states:")
    for i, discrete in enumerate(discrete_states):
        discrete = discrete_spaces[i].from_jsonable([discrete])[0]
        if not discrete_spaces[i].contains(discrete):
            print(f"Discrete state {i} does not match expected spaces.")
            print(f"Expected space: {discrete_spaces[i]}")
            print(f"Expected value: {discrete_actions[i]}, Received: {discrete}")
        else:
            print(f"Discrete state {i} matches expected state.")

    print("\nMulti-binary states:")
    for i, mb in enumerate(mb_states):
        mb = mb_spaces[i].from_jsonable([mb])[0]
        if not mb_spaces[i].contains(mb):
            print(f"Multi-binary state {i} does not match expected space.")
            print(f"Expected space: {mb_spaces[i]}")
            print(f"Expected value: {mb_actions[i]}, Received: {mb}")
        else:
            print(f"Multi-binary state {i} matches expected state.")

    print("\nMulti-discrete states:")
    for i, md in enumerate(md_states):
        md = md_spaces[i].from_jsonable([md])[0]
        if not md_spaces[i].contains(md):
            print(f"Multi-discrete state {i} does not match expected space.")
            print(f"Expected space: {md_spaces[i]}")
            print(f"Expected value: {md_actions[i]}, Received: {md}")
        else:
            print(f"Multi-discrete state {i} matches expected state.")

    print("\nText states:")
    for i, text in enumerate(text_states):
        text = text_spaces[i].from_jsonable([text])[0]
        if not text_spaces[i].contains(text):
            print(f"Text state {i} does not match expected space.")
            print(f"Expected space: {text_spaces[i]}")
            print(f"Expected value: {text_actions[i]}, Received: {text}")
        else:
            print(f"Text state {i} matches expected state.")

    
async def main():
    client = GamaSyncClient("localhost", 1001, other_message_handler=gama_server_message_handler)
    experiment_id = server_connection(client)
    print("\n------------------Test receive spaces------------------\n")
    test_receive_spaces(client, experiment_id)
    print("\n------------------Test send actions------------------\n")
    test_send_actions(client, experiment_id)
    print("\n------------------Test receive states------------------\n")
    test_receive_states(client, experiment_id)

if __name__ == "__main__":
    asyncio.run(main())