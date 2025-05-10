# Microevolution

Microevolution is a simulation-based AI learning game where an organism navigates a 2D environment to collect food, avoid moving obstacles, and optimize its survival strategy. The project includes a game core, a client for training and visualization, and a server for multiplayer contests.

## Project Structure

- **common**: Core game logic and settings, including the environment, organism, food, obstacles, and grid-based spatial management.
- **client**: Client-side application with options for:
  - **UI Training Simulation**: Interactive visualization of the training process (`python -m client.menu`).
  - **CLI Training**: Non-UI training mode for faster computation (`python -m client.trainer`).
- **server**: Server-side application enabling multiplayer contests between players (`python -m server.app`).

## Prerequisites

- Python >=3.11, <4.0
- [Poetry](https://python-poetry.org/) for dependency management

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/microevolution.git
   cd microevolution
   ```

2. Install Poetry (if not already installed):
   ```bash
   pip install poetry
   ```

3. Install dependencies using Poetry:
   ```bash
   poetry install
   ```
   This installs core dependencies (`numpy`, `torch`, `matplotlib`). To include client-specific dependencies (`pygame`, `pygame-menu`, `tensorboard`) or server-specific dependencies (`flask`, `flask-socketio`), use:
   ```bash
   poetry install --with client  # For client dependencies
   poetry install --with server  # For server dependencies
   ```

4. Activate the Poetry virtual environment:
   ```bash
   poetry shell
   ```

## Usage

### Running the Client

The client supports two modes: a UI-based training simulation and a CLI-based trainer.

- **UI Training Simulation**:
  Launch the interactive visualization:
  ```bash
  python -m client.menu
  ```
  This mode renders the simulation using Pygame, displaying the organism, food, obstacles, and real-time metrics like energy and rewards.

- **CLI Training**:
  Run training without a graphical interface for faster iterations:
  ```bash
  python -m client.trainer
  ```
  This mode is optimized for training AI models and logging performance metrics to the `output` directory.

### Running the Server

The server enables multiplayer contests where players can compete with their trained organisms.

- Start the server:
  ```bash
  python -m server.app
  ```
  The server uses Flask and Flask-SocketIO to manage client connections and contest logic. Ensure the server is running before clients attempt to connect.

## Game Overview

The microevolution simulation models an organism navigating a circular "dish" to collect food while avoiding moving obstacles. Key components include:

- **Organism**: Controlled by an AI agent using a reinforcement learning model, it moves based on actions (acceleration vectors), consumes energy, and gains energy by eating food.
- **Food**: Randomly spawned within the dish, food items replenish the organism's energy when consumed.
- **Obstacles**: Moving entities that the organism must avoid to prevent collisions and penalties.
- **Environment**: A 2D space with a grid for efficient spatial queries, a reward system, and a dish boundary.
- **Settings**: Configurable parameters for organism behavior, food spawn rates, obstacle properties, training hyperparameters, and general simulation settings.

The AI is trained using a reinforcement learning approach (actor-critic model) to optimize its movement strategy for maximizing food collection while minimizing energy loss and collisions.

## Settings

The project uses a modular settings structure shared across client and server, with specific configurations for training and rendering.

## Dependencies

The project uses Poetry for dependency management. Key dependencies include:

### Core Dependencies
- `python`: >=3.11, <4.0
- `numpy`: >=2.2.5, <3.0.0
- `torch`: >=2.7.0, <3.0.0
- `matplotlib`: >=3.10.3, <4.0.0

### Client Dependencies
- `pygame`: >=2.6.1
- `pygame-menu`: >=4.5.2
- `tensorboard`: >=2.17.0, <3.0.0

### Server Dependencies
- `flask`: >=3.1.0
- `flask-socketio`: >=5.5.1

See the `[tool.poetry]` section in `pyproject.toml` for the full configuration.

## Development

To contribute to the project:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Install dependencies with Poetry:
   ```bash
   poetry install --with client,server
   ```
4. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
5. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
6. Open a pull request.

## Contact

For questions or suggestions, open an issue on the GitHub repository or contact the maintainer at [artemgsr@gmail.com](mailto:artemgsr@gmail.com).