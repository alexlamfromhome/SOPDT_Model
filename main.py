from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
from SOPDT_Model import SOPDT_Model
import threading
import time

# Initialize FastAPI app
app = FastAPI(
    title="SOPDT Model API",
    description="API for controlling and monitoring a Second-Order Plus Dead Time (SOPDT) system with PID controller",
    version="1.0.0"
)

# Global model instance and control parameters
model: Optional[SOPDT_Model] = None
Kp = 1.0
Ki = 0.1
Kd = 0.05
setpoint = 0.0
simulation_running = False
simulation_thread = None

# Pydantic models for request/response
class ModelConfig(BaseModel):
    K: float
    tau: float
    zeta: float
    theta: float
    dt: float
    sigma1: float = 0.0
    sigma2: float = 0.0
    u_min: float = -np.inf
    u_max: float = np.inf

class PIDGains(BaseModel):
    Kp: float
    Ki: float
    Kd: float

class SetpointRequest(BaseModel):
    setpoint: float

class ModelState(BaseModel):
    x1: float
    x2: float
    Kp: float
    Ki: float
    Kd: float
    setpoint: float
    integral_error: float
    prev_error: float

# Simulation loop
def run_simulation(update_frequency: float = 10.0):
    """Run the model simulation loop at specified frequency (Hz)"""
    global model, Kp, Ki, Kd, setpoint, simulation_running
    
    dt_sim = 1.0 / update_frequency
    
    while simulation_running:
        if model is not None:
            model.step(setpoint, Kp, Ki, Kd)
        time.sleep(dt_sim)

# Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    # Default configuration
    model = SOPDT_Model(K=1.0, tau=1.0, zeta=0.7, theta=0.1, dt=0.01)

@app.post("/initialize")
async def initialize_model(config: ModelConfig):
    """
    Initialize or reinitialize the SOPDT model with custom parameters.
    
    Parameters:
    -----------
    K : float - Plant gain
    tau : float - Time constant
    zeta : float - Damping ratio
    theta : float - Dead time
    dt : float - Integration time step
    sigma1 : float - Noise intensity for position (optional)
    sigma2 : float - Noise intensity for velocity (optional)
    u_min : float - Minimum control saturation (optional)
    u_max : float - Maximum control saturation (optional)
    """
    global model
    try:
        model = SOPDT_Model(
            K=config.K,
            tau=config.tau,
            zeta=config.zeta,
            theta=config.theta,
            dt=config.dt,
            sigma1=config.sigma1,
            sigma2=config.sigma2,
            u_min=config.u_min if config.u_min != -np.inf else -np.inf,
            u_max=config.u_max if config.u_max != np.inf else np.inf
        )
        return {"status": "Model initialized successfully", "config": config}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to initialize model: {str(e)}")

@app.get("/state", response_model=ModelState)
async def get_state():
    """
    Get current state of the model and controller.
    
    Returns:
    --------
    x1 : float - Current process output (position)
    x2 : float - Current rate of change (velocity)
    Kp : float - Current proportional gain
    Ki : float - Current integral gain
    Kd : float - Current derivative gain
    setpoint : float - Current desired setpoint
    integral_error : float - Accumulated integral error
    prev_error : float - Previous error value
    """
    global model, Kp, Ki, Kd, setpoint
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return ModelState(
        x1=float(model.x1),
        x2=float(model.x2),
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        setpoint=setpoint,
        integral_error=float(model.integral_error),
        prev_error=float(model.prev_error)
    )

@app.post("/set_gains")
async def set_gains(gains: PIDGains):
    """
    Update PID controller gains.
    
    Parameters:
    -----------
    Kp : float - Proportional gain
    Ki : float - Integral gain
    Kd : float - Derivative gain
    """
    global Kp, Ki, Kd
    try:
        Kp = gains.Kp
        Ki = gains.Ki
        Kd = gains.Kd
        return {"status": "PID gains updated", "Kp": Kp, "Ki": Ki, "Kd": Kd}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set gains: {str(e)}")

@app.post("/set_setpoint")
async def set_setpoint(request: SetpointRequest):
    """
    Update the desired setpoint (reference value).
    
    Parameters:
    -----------
    setpoint : float - Desired reference value
    """
    global setpoint
    try:
        setpoint = request.setpoint
        return {"status": "Setpoint updated", "setpoint": setpoint}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set setpoint: {str(e)}")

@app.post("/start_simulation")
async def start_simulation(update_frequency: float = 10.0):
    """
    Start the simulation loop.
    
    Parameters:
    -----------
    update_frequency : float - Update frequency in Hz (default: 10 Hz)
    """
    global simulation_running, simulation_thread
    
    if simulation_running:
        return {"status": "Simulation already running"}
    
    simulation_running = True
    simulation_thread = threading.Thread(target=run_simulation, args=(update_frequency,), daemon=True)
    simulation_thread.start()
    
    return {"status": "Simulation started", "update_frequency_hz": update_frequency}

@app.post("/stop_simulation")
async def stop_simulation():
    """Stop the simulation loop."""
    global simulation_running
    
    if not simulation_running:
        return {"status": "Simulation not running"}
    
    simulation_running = False
    return {"status": "Simulation stopped"}

@app.get("/simulation_status")
async def get_simulation_status():
    """Get the current simulation status."""
    global simulation_running
    return {"simulation_running": simulation_running}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_initialized": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
