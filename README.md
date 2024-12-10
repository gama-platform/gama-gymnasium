# gama-gymnasium
A generic [gymnasium](https://gymnasium.farama.org/) environment to simulations from the modeling platform [GAMA](https://gama-platform.org/).
It leverages the python package [gama-client](https://pypi.org/project/gama-client/) to communicate with a [gama server](https://gama-platform.org/wiki/HeadlessServer) and provide the best performances.

Based on previous work available [here](https://github.com/ptaillandier/policy-design/), our goal is to provide a generic and efficient platform that could be reused and adapted by anyone instead of an ad-hoc solution.
 
## Components

The core of the project is composed of two parts:
 - The python `gama-gymnasium` package, stored in the `python_package/` directory. It contains a [gymnasium](https://gymnasium.farama.org/) (a fork of OpenAI's gym) [environment](https://gymnasium.farama.org/api/env/) able to manipulate the modeling platform [GAMA](https://gama-platform.org/) to do your reinforcement learning work on GAMA simulation.
 - Some GAMA components, stored in the `gama/` folder file that contains the base components needed in your model to be manipulated by the gymnasium environment.

In addition to this, examples are stored in the `examples/` directory, each example is itself composed of a complete GAMA model and its corresponding python reinforcement learning script.

## Requirements

For everything to work, you first need to have GAMA installed somewhere and able to run in headless mode.
You can check the GAMA [download page](https://gama-platform.org/download) or [installation page](https://gama-platform.org/wiki/Installation) for detailed explanation and more advanced installations.

Your python environment should also have the package `gama-client` installed:
```shell
pip install gama-client
```
And be able to install this plugin as will be explained later.

## How to run it

### Install the `gama-gymnasium` plugin

You will need to add gama-gymnasium to your package library.
#### From pypi
The easiest way to do is from the `pypi` package repository. To do so, just type:
```python
pip install gama-gymnasium
```
In your work environment to get the latest release.

#### From git

you can also get the most up-to-date version by installing the package directly from this code.
First you need to clone this repository:
```bash
git clone https://github.com/gama-platform/gama-gymnasium.git
```
Make sure you have the corresponding tools to build the package:
```
python -m pip install --upgrade build
```

Then, in the package folder run:
```
python -m build
```


#### Test

You can test that everything works properly by trying to run one of the model in the `examples` directory

### Adapt your GAMA model to be compatible

Add `gama gymnasium.gaml` to your project and create a species that will have `GymnasiumCommunication` as a parent.
It should look similar to this:
```
import '../path/to/gama gymnasium.gaml'

...

species GymnasiumManager parent:GymnasiumCommunication {

}
```

TODO: @ptaillandier

### Implements the `gamaEnv` class in your python RL project

TODO: @meritxell.vinyals

### Run GAMA in server mode

You will need to have GAMA running in server mode for this to work.
To do so, go to your gama installation folder, and in the `headless` folder run the script `gama-headless.sh` if you are on Linux or MacOS and `gama-headless.bat` if you are on Windows.
The command should look like this:
```shell
gama-headless.sh -socket 6868
```
the `-socket` parameter indicates that you want to run a GAMA server, and `6868` is just a random port number that will be used for the connexion, you can switch it to any other port you want.

Once it finished initialized, you can run your python script.