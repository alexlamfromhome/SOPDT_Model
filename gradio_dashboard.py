import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from SOPDT_Model import SOPDT_Model
import threading
import time
from collections import deque
from datetime import datetime

# Global state
model = SOPDT_Model(K=1.0, tau=1.0, zeta=0.7, theta=0.1, dt=0.01)
Kp = 1.0
Ki = 0.1
Kd = 0.05
setpoint = 1.0

# Data storage for plotting (keep last 500 samples)
max_history = 500
time_history = deque(maxlen=max_history)
output_history = deque(maxlen=max_history)
setpoint_history = deque(maxlen=max_history)
control_effort_history = deque(maxlen=max_history)
error_history = deque(maxlen=max_history)

simulation_running = False
simulation_thread = None
last_u = 0.0
sample_counter = 0

def run_simulation_loop(update_frequency=20):
    """Run the simulation in background"""
    global model, Kp, Ki, Kd, setpoint, simulation_running, last_u, sample_counter
    
    dt_sim = 1.0 / update_frequency
    
    while simulation_running:
        if model is not None:
            # Calculate control effort before step
            error = setpoint - model.x1
            P = Kp * error
            I = Ki * model.integral_error
            D = Kd * (error - model.prev_error) / model.dt
            u_unsaturated = P + I + D
            last_u = np.clip(u_unsaturated, -10, 10)
            
            # Step the model
            y = model.step(setpoint, Kp, Ki, Kd)
            
            # Store history
            time_history.append(sample_counter * dt_sim)
            output_history.append(float(y))
            setpoint_history.append(float(setpoint))
            control_effort_history.append(float(last_u))
            error_history.append(float(error))
            
            sample_counter += 1
        
        time.sleep(dt_sim)

def create_plots():
    """Create live plots of model outputs"""
    if len(time_history) == 0:
        # Return empty plot if no data
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=[], mode='lines', name='Empty'))
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Process Output vs Setpoint",
            "Control Effort",
            "Tracking Error",
            "System State"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    time_array = list(time_history)
    
    # Plot 1: Output and Setpoint
    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=list(output_history),
            mode='lines',
            name='Output (x1)',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=list(setpoint_history),
            mode='lines',
            name='Setpoint',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Plot 2: Control Effort
    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=list(control_effort_history),
            mode='lines',
            name='Control Effort (u)',
            line=dict(color='green', width=2),
            fill='tozeroy'
        ),
        row=1, col=2
    )
    
    # Plot 3: Error
    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=list(error_history),
            mode='lines',
            name='Error',
            line=dict(color='orange', width=2),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    # Plot 4: System states
    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=[model.x2] * len(time_array),  # Show current velocity
            mode='lines',
            name='Velocity (x2)',
            line=dict(color='purple', width=2)
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    
    fig.update_yaxes(title_text="Output", row=1, col=1)
    fig.update_yaxes(title_text="Control (u)", row=1, col=2)
    fig.update_yaxes(title_text="Error", row=2, col=1)
    fig.update_yaxes(title_text="Velocity", row=2, col=2)
    
    fig.update_layout(
        height=800,
        title_text="SOPDT Model Real-Time Monitoring",
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def update_gains(kp_val, ki_val, kd_val):
    """Update PID gains"""
    global Kp, Ki, Kd
    Kp = float(kp_val)
    Ki = float(ki_val)
    Kd = float(kd_val)
    return f"✓ Gains updated: Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.3f}"

def update_setpoint(sp_val):
    """Update setpoint"""
    global setpoint
    setpoint = float(sp_val)
    return f"✓ Setpoint updated to {setpoint:.3f}"

def start_sim():
    """Start simulation"""
    global simulation_running, simulation_thread
    if simulation_running:
        return "⚠ Simulation already running"
    
    simulation_running = True
    simulation_thread = threading.Thread(target=run_simulation_loop, args=(20,), daemon=True)
    simulation_thread.start()
    return "▶ Simulation started"

def stop_sim():
    """Stop simulation"""
    global simulation_running
    simulation_running = False
    return "⏹ Simulation stopped"

def reset_sim():
    """Reset model and data"""
    global model, time_history, output_history, setpoint_history, control_effort_history, error_history, sample_counter, simulation_running
    simulation_running = False
    time.sleep(0.5)
    
    model = SOPDT_Model(K=1.0, tau=1.0, zeta=0.7, theta=0.1, dt=0.01)
    time_history.clear()
    output_history.clear()
    setpoint_history.clear()
    control_effort_history.clear()
    error_history.clear()
    sample_counter = 0
    
    return "🔄 Model reset"

def get_stats():
    """Get current statistics"""
    if len(output_history) == 0:
        return "No data yet"
    
    output_arr = list(output_history)
    error_arr = list(error_history)
    
    return f"""
    **Current Statistics:**
    
    | Metric | Value |
    |--------|-------|
    | Current Output (x1) | {model.x1:.4f} |
    | Current Velocity (x2) | {model.x2:.4f} |
    | Current Error | {error_arr[-1]:.4f} |
    | Mean Output | {np.mean(output_arr):.4f} |
    | Std Dev Output | {np.std(output_arr):.4f} |
    | Min Output | {np.min(output_arr):.4f} |
    | Max Output | {np.max(output_arr):.4f} |
    | Mean Error | {np.mean(error_arr):.4f} |
    | Integral Error | {model.integral_error:.4f} |
    | Samples Collected | {len(output_history)} |
    """

# Create Gradio interface
with gr.Blocks(title="SOPDT PID Controller", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SOPDT Model PID Controller - Live Dashboard")
    gr.Markdown("Real-time monitoring and control of Second-Order Plus Dead Time system with PID controller")
    
    with gr.Row():
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Live System Response")
        
        with gr.Column(scale=1):
            gr.Markdown("### PID Gains Control")
            kp_slider = gr.Slider(
                minimum=0.0,
                maximum=5.0,
                value=1.0,
                step=0.01,
                label="Kp (Proportional)",
                info="Increase for faster response"
            )
            ki_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.1,
                step=0.01,
                label="Ki (Integral)",
                info="Eliminate steady-state error"
            )
            kd_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.05,
                step=0.01,
                label="Kd (Derivative)",
                info="Dampen oscillations"
            )
            gains_update_btn = gr.Button("Update Gains", variant="primary")
            gains_status = gr.Textbox(label="Status", value="Ready", interactive=False)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Setpoint Control")
            setpoint_slider = gr.Slider(
                minimum=-2.0,
                maximum=5.0,
                value=1.0,
                step=0.1,
                label="Reference Setpoint"
            )
            setpoint_update_btn = gr.Button("Set Setpoint", variant="primary")
            setpoint_status = gr.Textbox(label="Status", value="Ready", interactive=False)
        
        with gr.Column():
            gr.Markdown("### Simulation Control")
            with gr.Row():
                start_btn = gr.Button("▶ Start", variant="primary", size="lg")
                stop_btn = gr.Button("⏹ Stop", variant="stop", size="lg")
                reset_btn = gr.Button("🔄 Reset", size="lg")
            sim_status = gr.Textbox(label="Simulation Status", value="Ready", interactive=False)
    
    with gr.Row():
        stats_output = gr.Markdown("### System Statistics\nWaiting for simulation to start...")
    
    gr.Markdown("""
    ### Instructions
    1. Adjust **Kp, Ki, Kd** sliders to tune controller response
    2. Set desired **Setpoint** value
    3. Click **Start** to begin simulation
    4. Monitor live plots and statistics
    5. Adjust gains in real-time to see effects
    6. Click **Stop** to pause, **Reset** to clear data
    
    ### Tips
    - Start with low Kp and gradually increase
    - Add Ki to eliminate steady-state error
    - Use Kd to reduce overshoot and oscillations
    - Watch the plots to visualize system response
    """)
    
    # Set up event handlers
    gains_update_btn.click(
        fn=update_gains,
        inputs=[kp_slider, ki_slider, kd_slider],
        outputs=gains_status
    )
    
    setpoint_update_btn.click(
        fn=update_setpoint,
        inputs=setpoint_slider,
        outputs=setpoint_status
    )
    
    start_btn.click(
        fn=start_sim,
        outputs=sim_status
    )
    
    stop_btn.click(
        fn=stop_sim,
        outputs=sim_status
    )
    
    reset_btn.click(
        fn=reset_sim,
        outputs=sim_status
    )
    
    # Auto-update plot and stats
    demo.load(
        fn=lambda: (create_plots(), get_stats()),
        outputs=[plot_output, stats_output],
        every=0.5
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
