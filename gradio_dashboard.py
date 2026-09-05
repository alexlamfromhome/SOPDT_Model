import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import threading
import time
from collections import deque

# FastAPI backend URL
API_BASE_URL = "http://localhost:8000"

# Data storage for plotting (keep last 500 samples)
max_history = 500
time_history = deque(maxlen=max_history)
output_history = deque(maxlen=max_history)
setpoint_history = deque(maxlen=max_history)
control_effort_history = deque(maxlen=max_history)
error_history = deque(maxlen=max_history)
velocity_history = deque(maxlen=max_history)

monitoring_thread = None
monitoring_running = False
sample_counter = 0

def check_api_health():
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def monitor_model():
    """Monitor model state from FastAPI backend"""
    global sample_counter, monitoring_running
    
    while monitoring_running:
        try:
            # Get current state from API
            response = requests.get(f"{API_BASE_URL}/state", timeout=2)
            if response.status_code == 200:
                state = response.json()
                
                # Store history
                time_history.append(sample_counter * 0.05)  # 50ms intervals
                output_history.append(float(state['x1']))
                setpoint_history.append(float(state['setpoint']))
                velocity_history.append(float(state['x2']))
                
                # Calculate error
                error = state['setpoint'] - state['x1']
                error_history.append(float(error))
                
                # Estimate control effort (simplified)
                P = state['Kp'] * error
                I = state['Ki'] * state['integral_error']
                D = state['Kd'] * state['prev_error']
                u = np.clip(P + I + D, -10, 10)
                control_effort_history.append(float(u))
                
                sample_counter += 1
            
            time.sleep(0.05)  # 20 Hz monitoring
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(0.5)

def create_plots():
    """Create live plots of model outputs"""
    if len(time_history) == 0:
        # Return empty plot if no data
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=[], mode='lines', name='Empty'))
        fig.update_layout(title="Waiting for data...", height=800)
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Process Output vs Setpoint",
            "Control Effort",
            "Tracking Error",
            "System Velocity"
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
    
    # Plot 4: Velocity
    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=list(velocity_history),
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
        title_text="SOPDT Model Real-Time Monitoring (via FastAPI)",
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def update_gains(kp_val, ki_val, kd_val):
    """Update PID gains via FastAPI"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/set_gains",
            json={"Kp": float(kp_val), "Ki": float(ki_val), "Kd": float(kd_val)},
            timeout=2
        )
        if response.status_code == 200:
            data = response.json()
            return f"✓ Gains updated: Kp={data['Kp']:.3f}, Ki={data['Ki']:.3f}, Kd={data['Kd']:.3f}"
        else:
            return "✗ Failed to update gains"
    except Exception as e:
        return f"✗ Error: {str(e)}"

def update_setpoint(sp_val):
    """Update setpoint via FastAPI"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/set_setpoint",
            json={"setpoint": float(sp_val)},
            timeout=2
        )
        if response.status_code == 200:
            data = response.json()
            return f"✓ Setpoint updated to {data['setpoint']:.3f}"
        else:
            return "✗ Failed to update setpoint"
    except Exception as e:
        return f"✗ Error: {str(e)}"

def start_sim():
    """Start simulation via FastAPI"""
    global monitoring_running, monitoring_thread
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/start_simulation?update_frequency=20",
            timeout=2
        )
        if response.status_code == 200:
            if not monitoring_running:
                monitoring_running = True
                monitoring_thread = threading.Thread(target=monitor_model, daemon=True)
                monitoring_thread.start()
            data = response.json()
            return f"▶ {data['status']} at {data['update_frequency_hz']} Hz"
        else:
            return "✗ Failed to start simulation"
    except Exception as e:
        return f"✗ Error: {str(e)}"

def stop_sim():
    """Stop simulation via FastAPI"""
    global monitoring_running
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/stop_simulation",
            timeout=2
        )
        monitoring_running = False
        if response.status_code == 200:
            data = response.json()
            return f"⏹ {data['status']}"
        else:
            return "✗ Failed to stop simulation"
    except Exception as e:
        return f"✗ Error: {str(e)}"

def reset_sim():
    """Reset model and data"""
    global time_history, output_history, setpoint_history, control_effort_history, error_history, velocity_history, sample_counter, monitoring_running
    
    try:
        monitoring_running = False
        time.sleep(0.5)
        
        time_history.clear()
        output_history.clear()
        setpoint_history.clear()
        control_effort_history.clear()
        error_history.clear()
        velocity_history.clear()
        sample_counter = 0
        
        return "🔄 Model and data reset"
    except Exception as e:
        return f"✗ Error: {str(e)}"

def get_stats():
    """Get current statistics from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/state", timeout=2)
        if response.status_code == 200:
            state = response.json()
            
            if len(output_history) == 0:
                return "Waiting for data..."
            
            output_arr = list(output_history)
            error_arr = list(error_history)
            
            return f"""
    **Current Statistics (Live from FastAPI):**
    
    | Metric | Value |
    |--------|-------|
    | Current Output (x1) | {state['x1']:.4f} |
    | Current Velocity (x2) | {state['x2']:.4f} |
    | Current Error | {state['setpoint'] - state['x1']:.4f} |
    | Mean Output | {np.mean(output_arr):.4f} |
    | Std Dev Output | {np.std(output_arr):.4f} |
    | Min Output | {np.min(output_arr):.4f} |
    | Max Output | {np.max(output_arr):.4f} |
    | Mean Error | {np.mean(error_arr):.4f} |
    | Integral Error | {state['integral_error']:.4f} |
    | Current Gains | Kp={state['Kp']:.3f}, Ki={state['Ki']:.3f}, Kd={state['Kd']:.3f} |
    | Samples Collected | {len(output_history)} |
    """
        else:
            return "✗ Failed to get statistics"
    except Exception as e:
        return f"✗ Error connecting to API: {str(e)}"

def get_api_status():
    """Check API health"""
    if check_api_health():
        return "✓ FastAPI Backend: Connected"
    else:
        return "✗ FastAPI Backend: Disconnected (Start main.py first)"

# Create Gradio interface
with gr.Blocks(title="SOPDT PID Controller", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SOPDT Model PID Controller - Live Dashboard")
    gr.Markdown("Real-time monitoring and control via FastAPI backend")
    
    api_status = gr.Textbox(label="API Status", interactive=False, scale=1)
    
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
        stats_output = gr.Markdown("### System Statistics\nWaiting for connection...")
    
    gr.Markdown("""
    ### Instructions
    1. **Start FastAPI Backend**: Run `python main.py` in another terminal
    2. Adjust **Kp, Ki, Kd** sliders to tune controller response
    3. Set desired **Setpoint** value
    4. Click **Start** to begin simulation
    5. Monitor live plots and statistics in real-time
    6. Adjust gains in real-time to see effects
    7. Click **Stop** to pause, **Reset** to clear data
    
    ### Tips
    - Start with low Kp and gradually increase
    - Add Ki to eliminate steady-state error
    - Use Kd to reduce overshoot and oscillations
    - Watch the plots to visualize system response
    - All adjustments are sent to the FastAPI backend
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
    
    # Auto-update plot, stats, and API status
    demo.load(
        fn=lambda: (get_api_status(), create_plots(), get_stats()),
        outputs=[api_status, plot_output, stats_output],
        every=1
    )

if __name__ == "__main__":
    print("Starting Gradio Dashboard...")
    print("Make sure FastAPI backend is running: python main.py")
    print("Gradio will be available at http://localhost:7860")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
