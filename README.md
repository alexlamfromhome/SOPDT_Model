# SOPDT Model with PID Controller

A Python implementation of a **Second-Order Plus Dead Time (SOPDT)** system with an integrated **PID controller** featuring anti-windup control and stochastic dynamics. Includes a **FastAPI** frontend for easy interaction and monitoring.

## Overview

This project simulates a closed-loop control system with the following features:

- **SOPDT Plant Dynamics**: Models a second-order system with:
  - Proportional gain (K)
  - Time constant (tau)
  - Damping ratio (zeta)
  - Dead time delay (theta)
  - Stochastic noise injection (Euler-Maruyama integration)

- **PID Controller**: Classic three-term controller with:
  - Proportional (P) term for immediate error response
  - Integral (I) term for steady-state error elimination
  - Derivative (D) term for rate damping
  - **Anti-windup** back-calculation to prevent integral saturation
  - Control effort saturation limits

- **FastAPI REST API**: Web-based interface for:
  - Querying real-time model state
  - Adjusting PID gains dynamically
  - Setting reference setpoints
  - Starting/stopping simulations
  - Configuring model parameters

## Installation

### Requirements
- Python 3.8+
- numpy
- fastapi
- uvicorn

### Setup

```bash
# Clone the repository
git clone https://github.com/alexlamfromhome/SOPDT_Model.git
cd SOPDT_Model

# Install dependencies
pip install numpy fastapi uvicorn
```

## Usage

### Running the API Server

```bash
python main.py
```

The API will start on `http://localhost:8000`

Access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Python Example

```python
from SOPDT_Model import SOPDT_Model

# Initialize model
model = SOPDT_Model(
    K=1.0,      # Plant gain
    tau=1.0,    # Time constant
    zeta=0.7,   # Damping ratio
    theta=0.1,  # Dead time
    dt=0.01,    # Integration step
    sigma1=0.0, # Position noise
    sigma2=0.0, # Velocity noise
    u_min=-10,  # Min control effort
    u_max=10    # Max control effort
)

# Simulate with PID gains
Kp, Ki, Kd = 1.0, 0.1, 0.05
setpoint = 1.0

for _ in range(1000):
    y = model.step(setpoint, Kp, Ki, Kd)
    print(f"Output: {y:.4f}")
```

## API Endpoints

### Health & Status

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_initialized": true
}
```

---

#### `GET /simulation_status`
Get current simulation running status.

**Response:**
```json
{
  "simulation_running": true
}
```

---

### Model State

#### `GET /state`
Query current state of the model and controller.

**Response:**
```json
{
  "x1": 0.5234,           // Current process output (position)
  "x2": 0.0123,           // Current rate of change (velocity)
  "Kp": 1.0,              // Proportional gain
  "Ki": 0.1,              // Integral gain
  "Kd": 0.05,             // Derivative gain
  "setpoint": 1.0,        // Current desired setpoint
  "integral_error": 0.234, // Accumulated integral error
  "prev_error": 0.45      // Previous error value
}
```

---

### Configuration

#### `POST /initialize`
Initialize or reinitialize the SOPDT model with custom parameters.

**Request Body:**
```json
{
  "K": 1.0,           // Plant gain
  "tau": 1.0,         // Time constant
  "zeta": 0.7,        // Damping ratio
  "theta": 0.1,       // Dead time (seconds)
  "dt": 0.01,         // Integration time step (seconds)
  "sigma1": 0.0,      // Noise intensity for position (optional)
  "sigma2": 0.0,      // Noise intensity for velocity (optional)
  "u_min": -10.0,     // Minimum control saturation (optional)
  "u_max": 10.0       // Maximum control saturation (optional)
}
```

**Response:**
```json
{
  "status": "Model initialized successfully",
  "config": { /* echoed config */ }
}
```

---

### PID Tuning

#### `POST /set_gains`
Update PID controller gains dynamically.

**Request Body:**
```json
{
  "Kp": 1.2,
  "Ki": 0.15,
  "Kd": 0.08
}
```

**Response:**
```json
{
  "status": "PID gains updated",
  "Kp": 1.2,
  "Ki": 0.15,
  "Kd": 0.08
}
```

---

#### `POST /set_setpoint`
Update the desired reference setpoint.

**Request Body:**
```json
{
  "setpoint": 2.5
}
```

**Response:**
```json
{
  "status": "Setpoint updated",
  "setpoint": 2.5
}
```

---

### Simulation Control

#### `POST /start_simulation`
Start the simulation loop.

**Query Parameters:**
- `update_frequency` (float, optional): Update frequency in Hz (default: 10 Hz)

**Example:**
```
POST /start_simulation?update_frequency=20
```

**Response:**
```json
{
  "status": "Simulation started",
  "update_frequency_hz": 20
}
```

---

#### `POST /stop_simulation`
Stop the simulation loop.

**Response:**
```json
{
  "status": "Simulation stopped"
}
```

---

## Example Workflows

### Workflow 1: Initialize and Run Simulation

```bash
# 1. Initialize model with custom parameters
curl -X POST "http://localhost:8000/initialize" \
  -H "Content-Type: application/json" \
  -d '{
    "K": 2.0,
    "tau": 0.5,
    "zeta": 0.6,
    "theta": 0.05,
    "dt": 0.01,
    "u_min": -5,
    "u_max": 5
  }'

# 2. Set PID gains
curl -X POST "http://localhost:8000/set_gains" \
  -H "Content-Type: application/json" \
  -d '{"Kp": 1.5, "Ki": 0.2, "Kd": 0.1}'

# 3. Set setpoint
curl -X POST "http://localhost:8000/set_setpoint" \
  -H "Content-Type: application/json" \
  -d '{"setpoint": 1.0}'

# 4. Start simulation at 50 Hz
curl -X POST "http://localhost:8000/start_simulation?update_frequency=50"

# 5. Query state in a loop
for i in {1..10}; do
  curl -X GET "http://localhost:8000/state"
  sleep 0.1
done

# 6. Stop simulation
curl -X POST "http://localhost:8000/stop_simulation"
```

### Workflow 2: Dynamic PID Tuning

```bash
# Start simulation
curl -X POST "http://localhost:8000/start_simulation?update_frequency=20"

# Adjust gains while running
for Kp in 0.5 1.0 1.5 2.0; do
  curl -X POST "http://localhost:8000/set_gains" \
    -H "Content-Type: application/json" \
    -d "{\"Kp\": $Kp, \"Ki\": 0.1, \"Kd\": 0.05}"
  sleep 2
done

# Stop
curl -X POST "http://localhost:8000/stop_simulation"
```

## Architecture

### SOPDT_Model Class

The core dynamics are governed by:

```
dx1/dt = x2
dx2/dt = -(1/τ²)·x1 - (2ζ/τ)·x2 + (K/τ²)·u_delayed + noise
```

Where:
- `x1`: Process output (position)
- `x2`: Process rate (velocity)
- `u_delayed`: Control effort after dead time delay
- Noise: Euler-Maruyama stochastic integration

### PID Controller with Anti-Windup

```
P = Kp · error
I = Ki · ∫error dt (with anti-windup correction)
D = Kd · d(error)/dt
u = P + I + D (saturated to [u_min, u_max])
```

**Anti-Windup Mechanism:**
When the control effort saturates, the integral term is corrected by:
```
I_corrected = I + (u_saturated - u_unsaturated) / Ki
```

This prevents the integral error from accumulating when the actuator can't move further.

## Model Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---|
| **K** | Plant DC gain | 0.1 - 10 |
| **tau** | Time constant (seconds) | 0.01 - 10 |
| **zeta** | Damping ratio | 0.1 - 2.0 |
| **theta** | Dead time delay (seconds) | 0 - 1.0 |
| **dt** | Integration step (seconds) | 0.001 - 0.1 |
| **sigma1** | Position noise intensity | 0 - 1.0 |
| **sigma2** | Velocity noise intensity | 0 - 1.0 |
| **u_min** | Min control saturation | -∞ or negative |
| **u_max** | Max control saturation | +∞ or positive |

## PID Tuning Tips

- **Kp (Proportional)**: Increase for faster response, decrease if oscillating
- **Ki (Integral)**: Increase to eliminate steady-state error, decrease if overshooting
- **Kd (Derivative)**: Increase to dampen oscillations, decrease if noisy

Start with small values and incrementally increase until desired performance is achieved.

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.
