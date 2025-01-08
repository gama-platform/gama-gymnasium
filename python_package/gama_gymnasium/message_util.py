"""
This file contains a collection of functions dedicated to handling messages between
the gymnasium environment and a gama simulation.
"""


def observation_contains_end(observations: str) -> bool :
    """
    Reads a message containing observations and check if it contains the end of simulation signal.
    :param observations: the message containing the observations
    :return: True if the observations contained an end of simulation message
    """
    end = "END" in observations
    if end:
        observations = observations.replace("END", "")
    return end


def string_to_array(array_as_string: str) -> list[float]:
    """
    Converts a string to a numpy array of floats
    :param array_as_string: an array represented in a string
    :return: a numpy array
    """
    # first we remove brackets and parentheses
    clean = "".join([c if c not in "()[]{}" else '' for c in str(array_as_string)])
    # then we split into numbers
    nbs = [float(nb) for nb in filter(lambda s: s.strip() != "", clean.split(','))]
    return nbs


def action_to_string(actions: list) -> str:
    """
    Converts an action to a string to be sent to the simulation
    :param actions: an array representing the actions to send
    :return: a string that could be read as an action by the simulation
    """
    return ",".join([str(action) for action in actions]) + "\n"

