# gama-gymnasium
A generic [gymnasium](https://gymnasium.farama.org/) environment to manipulate simulations from the modeling platform [GAMA](https://gama-platform.org/).
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

Then, in the projects's folder run:
```
pip install -e python_package
```

#### Test the installation of the plugin

You should now be able to import `gama_gymnasium` in your projects. To test it, in your python environment run:
```
import gama_gymnasium
```
Normally you shouldn't see any error message. If it's the case check that you have installed the packages mentioned in the requirements section.

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

You will need to have GAMA running in server mode for the communication to work.
To do so, go to your gama installation folder, and in the `headless` folder run the script `gama-headless.sh` if you are on Linux or MacOS and `gama-headless.bat` if you are on Windows.
The command should look like this:
```shell
gama-headless.sh -socket 6868
```
the `-socket` parameter indicates that you want to run a GAMA server, and `6868` is just a random port number that will be used for the connexion, you can switch it to any other port you want.

Once it finished initialized, you can run your python script.

### Test the whole pipeline

You can go in the `tests` folder and in the `gamaenv_loads_simulation.py` and in the code change the value of `gama_port` to the port number you set yourself for gama-server in the previous section.
Once it's done you can run this python script.
If everything works you should see a few messages allocating a communication port in the console and then the program hang forever.

## Development & Testing

### Test Suite

GAMA-Gymnasium comes with a comprehensive test suite that ensures the reliability and quality of the package. The tests are organized into several categories:

#### Unit Tests
- **Space Converters** (`test_space_converters.py`): Tests for converting between GAMA and Gymnasium space formats
- **Space Validators** (`test_space_validators.py`): Tests for validating spaces and actions
- **Client Communication** (`test_client.py`): Tests for GAMA server communication
- **Message Handling** (`test_message_handler.py`): Tests for message validation and formatting
- **Synchronous Wrapper** (`test_sync_wrapper.py`): Tests for async/sync conversion
- **Monitoring Wrapper** (`test_monitoring_wrapper.py`): Tests for episode tracking and logging

#### Integration Tests
- **Full Workflow** (`test_full_integration.py`): End-to-end testing of the complete GAMA-Gymnasium pipeline

#### Performance Tests
- **Benchmarks** (`test_performance.py`): Performance testing and optimization validation

### Running Tests

To run the test suite, you'll need to install the development dependencies:

```bash
pip install -e ".[dev]"
```

Then you can run tests using the provided test runner:

```bash
# Run all tests
python run_tests.py all

# Run only unit tests
python run_tests.py unit

# Run fast tests (excluding slow performance tests)
python run_tests.py fast

# Check test dependencies
python run_tests.py deps
```

Or using Make (Windows compatible):

```bash
# Run all tests with coverage
make test

# Run unit tests only
make test-unit

# Check code formatting and linting
make lint
```

For detailed testing documentation, see [tests/README.md](tests/README.md).

### Contributing

When contributing to the project, please ensure:

1. All tests pass: `python run_tests.py all`
2. Code is properly formatted: `python run_tests.py format`
3. No linting errors: `python run_tests.py lint`
4. New features include appropriate tests

The test suite includes comprehensive coverage of space conversion functionality, which is critical for ensuring compatibility between GAMA simulation environments and Gymnasium's standardized space definitions.
