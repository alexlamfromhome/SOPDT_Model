import numpy as np
from collections import deque

class SOPDT_Model:
    def __init__(self, K, tau, zeta, theta, dt, sigma1=0.0, sigma2=0.0, u_min=-np.inf, u_max=np.inf):
        """
        Closed-Loop Second-Order Plus Dead Time (SOPDT) Model 
        with internal PID controller, anti-windup, and Euler-Maruyama integration.
        
        Parameters:
        -----------
        K : float
            Plant gain
        tau : float
            Time constant
        zeta : float
            Damping ratio
        theta : float
            Dead time
        dt : float
            Integration time step
        sigma1 : float, optional
            Noise intensity for x1 (default: 0.0)
        sigma2 : float, optional
            Noise intensity for x2 (default: 0.0)
        u_min : float, optional
            Minimum control effort saturation limit (default: -inf)
        u_max : float, optional
            Maximum control effort saturation limit (default: +inf)
        """
        self.K = K
        self.tau = tau
        self.zeta = zeta
        self.dt = dt
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.u_min = u_min
        self.u_max = u_max
        
        # Initialize plant state variables
        self.x1 = 0.0  # Position (Output / Process Variable)
        self.x2 = 0.0  # Velocity (Rate of change)
        
        # Initialize PID controller states
        self.integral_error = 0.0
        self.prev_error = 0.0
        
        # Dead time ring buffer for the control effort (u)
        self.delay_steps = max(0, int(round(theta / dt)))
        self.u_buffer = deque([0.0] * self.delay_steps, maxlen=max(1, self.delay_steps))

    def step(self, setpoint, Kp, Ki, Kd):
        """
        Computes the PID control effort with anti-windup and updates the plant state.
        
        Parameters:
        -----------
        setpoint : float
            Desired reference value
        Kp : float
            Proportional gain
        Ki : float
            Integral gain
        Kd : float
            Derivative gain
            
        Returns:
        --------
        y : float
            Plant output (process variable)
        """
        # --- 1. PID Controller Logic ---
        # Calculate current error (Setpoint - Process Variable)
        error = setpoint - self.x1
        
        # Proportional term
        P = Kp * error
        
        # Integral term (Euler integration) with anti-windup
        self.integral_error += error * self.dt
        I = Ki * self.integral_error
        
        # Derivative term (Backward difference)
        derivative = (error - self.prev_error) / self.dt
        D = Kd * derivative
        
        # Store current error for the next step's derivative calculation
        self.prev_error = error
        
        # Total control effort (unsaturated)
        u_unsaturated = P + I + D
        
        # --- 2. Saturation ---
        u_current = np.clip(u_unsaturated, self.u_min, self.u_max)
        
        # --- 3. Anti-Windup (Back-Calculation Method) ---
        # If saturation occurred, reduce the integral term
        if u_current != u_unsaturated:
            # Calculate the saturation error
            saturation_error = u_current - u_unsaturated
            # Back-calculate: reduce the integral state
            # This prevents the integral from growing when saturated
            self.integral_error += saturation_error / Ki if Ki != 0 else 0
        
        # --- 4. Plant Dead Time Delay ---
        if self.delay_steps > 0:
            u_delayed = self.u_buffer[0]
            self.u_buffer.append(u_current)
        else:
            u_delayed = u_current
            
        # --- 5. Plant Euler-Maruyama Integration ---
        # Generate standard normal noise for Euler-Maruyama
        w1_k = np.random.normal(0, 1)
        w2_k = np.random.normal(0, 1)
        
        # Compute continuous derivatives based on current states
        dx1 = self.x2
        dx2 = (-1 / (self.tau**2)) * self.x1 \
            - (2 * self.zeta / self.tau) * self.x2 \
            + (self.K / (self.tau**2)) * u_delayed
            
        # Compute next state 
        self.x1 = self.x1 + self.dt * dx1 + self.sigma1 * np.sqrt(self.dt) * w1_k
        self.x2 = self.x2 + self.dt * dx2 + self.sigma2 * np.sqrt(self.dt) * w2_k
        
        # Output is the current position/value
        y = self.x1
        
        return y
