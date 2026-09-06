import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import threading
import time
from collections import deque

from SOPDT_Model import Kp, Ki, Kd, setpoint, u_min, u_max

# FastAPI backend URL
API_BASE_URL = "http://localhost:8000"

# Data storage for plotting (keep 5 minutes at 20 Hz)
max_history = 6000
time_history = deque(maxlen=max_history)
output_history = deque(maxlen=max_history)
setpoint_history = deque(maxlen=max_history)
control_effort_history = deque(maxlen=max_history)
error_history = deque(maxlen=max_history)

monitoring_thread = None
monitoring_running = False
sample_counter = 0


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
                output_history.append(float(state["x1"]))
                setpoint_history.append(float(state["setpoint"]))

                # Calculate error
                error = state["setpoint"] - state["x1"]
                error_history.append(float(error))

                # Estimate control effort (simplified)
                P = state["Kp"] * error
                I = state["Ki"] * state["integral_error"]
                D = state["Kd"] * state["prev_error"]
                u = np.clip(P + I + D, u_min, u_max)
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
        fig.add_trace(go.Scatter(y=[], mode="lines", name="Empty"))
        fig.update_layout(title="Waiting for data...", height=800)
        return fig

    # The process response spans the full width; the supporting signals share
    # the second row.
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Process Output vs Setpoint",
            "Control Effort",
            "Tracking Error",
        ),
        specs=[
            [{"secondary_y": False, "colspan": 2}, None],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    time_array = list(time_history)

    # Plot 1: Output and Setpoint
    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=list(output_history),
            mode="lines",
            name="Output (x1)",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=list(setpoint_history),
            mode="lines",
            name="Setpoint",
            line=dict(color="red", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=list(control_effort_history),
            mode="lines",
            name="Control Effort (u)",
            line=dict(color="green", width=2),
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=time_array,
            y=list(error_history),
            mode="lines",
            name="Error",
            line=dict(color="orange", width=2),
            fill="tozeroy",
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    fig.update_yaxes(title_text="Output", row=1, col=1)
    fig.update_yaxes(title_text="Control (u)", row=2, col=1)
    fig.update_yaxes(title_text="Error", row=2, col=2)

    fig.update_layout(
        height=500,
        title_text="SOPDT Model Real-Time Monitoring (via FastAPI)",
        showlegend=True,
        hovermode="x unified",
        legend=dict(xanchor="center", yanchor="bottom", orientation="h", x=0.5, y=-0.2),
    )

    return fig


def update_gains(kp_val, ki_val, kd_val):
    """Update PID gains via FastAPI"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/set_gains",
            json={"Kp": float(kp_val), "Ki": float(ki_val), "Kd": float(kd_val)},
            timeout=2,
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
            f"{API_BASE_URL}/set_setpoint", json={"setpoint": float(sp_val)}, timeout=2
        )
        if response.status_code == 200:
            data = response.json()
            return f"✓ Setpoint updated to {data['setpoint']:.3f}"
        else:
            return "✗ Failed to update setpoint"
    except Exception as e:
        return f"✗ Error: {str(e)}"


def refresh_dashboard():
    """Refresh the live plot."""
    return create_plots()


def initialize_dashboard():
    """Start dashboard monitoring when the page opens."""
    global monitoring_running, monitoring_thread

    if not monitoring_running:
        monitoring_running = True
        monitoring_thread = threading.Thread(target=monitor_model, daemon=True)
        monitoring_thread.start()

    return create_plots()


# Create Gradio interface
with gr.Blocks(title="SOPDT PID Controller") as demo:
    gr.Markdown("# SOPDT Model PID Controller - Live Dashboard")

    plot_output = gr.Plot(label="Live System Response")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Setpoint")
            setpoint_slider = gr.Slider(
                minimum=-10.0,
                maximum=50.0,
                value=setpoint,
                step=1.0,
                label="Reference setpoint",
            )

        with gr.Column():
            gr.Markdown("### PID Gains")
            kp_slider = gr.Slider(
                minimum=0.0,
                maximum=40.0,
                value=Kp,
                step=0.01,
                label="Kp (Proportional)",
            )
            ki_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=Ki,
                step=0.01,
                label="Ki (Integral)",
            )
            kd_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=Kd,
                step=0.01,
                label="Kd (Derivative)",
            )

    # Update controls after the user releases each slider.
    kp_slider.release(
        fn=update_gains,
        inputs=[kp_slider, ki_slider, kd_slider],
    )
    ki_slider.release(
        fn=update_gains,
        inputs=[kp_slider, ki_slider, kd_slider],
    )
    kd_slider.release(
        fn=update_gains,
        inputs=[kp_slider, ki_slider, kd_slider],
    )

    setpoint_slider.release(
        fn=update_setpoint,
        inputs=setpoint_slider,
    )

    # Start monitoring immediately, then refresh the live plot once per second.
    demo.load(
        fn=initialize_dashboard,
        outputs=plot_output,
    )
    dashboard_timer = gr.Timer(1)
    dashboard_timer.tick(fn=refresh_dashboard, outputs=plot_output)

if __name__ == "__main__":
    print("Start the merged FastAPI and Gradio application with: python main.py")
