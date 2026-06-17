import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import time

# Set page config
st.set_page_config(layout="wide", page_title="Episodic Meta-LLM - Control Center")

def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        return json.load(f)

demo1_path = "MADRL/simulations/output/demo1_disruption.json"
demo2_path = "MADRL/simulations/output/demo2_alpha_impact.json"

demo1_data = load_data(demo1_path)
demo2_data = load_data(demo2_path)

if not demo1_data or not demo2_data:
    st.error("**Demo data not found. Run demo1_disruption.py and demo2_alpha_impact.py first.**")
    st.stop()

agent_names = ["Retailer", "Warehouse", "Distributor", "Manufacturer"]
colors = px.colors.qualitative.Plotly

def get_status_and_color(inv, back):
    if back == 0:
        return "Healthy", "#d4edda", "#155724" # Green
    ratio = inv / max(1.0, back)
    if ratio < 0.2:
        return "Stock Starved", "#f8d7da", "#721c24" # Red
    elif ratio > 5.0:
        return "Inventory Heavy", "#cce5ff", "#004085" # Blue
    else:
        return "Warning", "#fff3cd", "#856404" # Yellow

def render_sc_diagram(statuses):
    html = "<div style='display: flex; align-items: center; justify-content: space-between; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-bottom: 20px; font-weight: bold;'>"
    
    # Customer is always neutral
    html += f"<div style='padding: 10px 20px; border-radius: 5px; background-color: #e2e3e5; color: #383d41;'>Customer</div>"
    
    for i, name in enumerate(agent_names):
        html += "<div style='font-size: 24px;'>➔</div>"
        bg_col, txt_col = statuses[i][1], statuses[i][2]
        html += f"<div style='padding: 10px 20px; border-radius: 5px; background-color: {bg_col}; color: {txt_col}; text-align: center;'>{name}<br><span style='font-size: 12px; font-weight: normal;'>{statuses[i][0]}</span></div>"
        
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

st.title("Episodic Meta-LLM — Digital Twin Control Center")

st.sidebar.header("Data Status")
st.sidebar.success(f"✅ Demo 1 & Demo 2 Data Loaded")

tab1, tab2 = st.tabs(["Disruption Resilience (Demo 1)", "Alpha Impact (Demo 2)"])

# --- TAB 1: DISRUPTION RESILIENCE ---
with tab1:
    st.header("Interactive Replay")
    
    timesteps = demo1_data["timesteps"]
    max_t = len(timesteps) - 1
    
    if "replay_t" not in st.session_state:
        st.session_state.replay_t = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False
        
    # Play Controls
    col_play, col_pause, col_next, col_slider = st.columns([1, 1, 1, 7])
    with col_play:
        if st.button("▶️ Play (Slow)", use_container_width=True):
            st.session_state.playing = True
            st.rerun()
    with col_pause:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.playing = False
            st.rerun()
    with col_next:
        if st.button("⏭️ Next", use_container_width=True):
            if st.session_state.replay_t < max_t:
                st.session_state.replay_t += 1
            st.session_state.playing = False
            st.rerun()
            
    with col_slider:
        t = st.slider("Timestep", 0, max_t, st.session_state.replay_t, key="slider_t", label_visibility="collapsed")
        st.session_state.replay_t = t
        
    ts = timesteps[t]
    
    # Check for special events
    if t >= 80 and t < 100:
        st.error("🚨 DISRUPTION ACTIVE: Manufacturer production halted.")
    elif t == 100:
        st.success("✅ DISRUPTION ENDS: Manufacturer resumes production.")
    elif t >= 80 + demo1_data["summary"].get("recovery_time", 0) and t > 80:
        st.info("🎉 SYSTEM RECOVERED: Service level returned to baseline.")
        
    # Calculate status
    agent_status = []
    invs, backs, ords = [], [], []
    for i, name in enumerate(agent_names):
        key = name.lower()
        inv = ts['inventory'].get(key, ts['inventory'].get(str(i), 0)) if isinstance(ts['inventory'], dict) else ts['inventory'][i]
        back = ts['backlog'].get(key, ts['backlog'].get(str(i), 0)) if isinstance(ts['backlog'], dict) else ts['backlog'][i]
        ord_q = ts['order_qty'].get(key, ts['order_qty'].get(str(i), 0)) if isinstance(ts['order_qty'], dict) else ts['order_qty'][i]
        
        invs.append(inv)
        backs.append(back)
        ords.append(ord_q)
        status_text, bg, fg = get_status_and_color(inv, back)
        agent_status.append((status_text, bg, fg))
        
    render_sc_diagram(agent_status)
    
    # System KPI Row
    st.subheader("CURRENT SYSTEM STATE")
    kpi_col1, kpi_col2, _ = st.columns([2, 2, 6])
    with kpi_col1:
        st.metric("Rolling Service Level", f"{ts.get('service_level_rolling', 0):.3f}")
    with kpi_col2:
        st.metric("Cumulative Cost", f"${ts.get('cumulative_cost', 0):,.0f}")
        
    st.markdown("---")
        
    cols = st.columns(4)
    for i, name in enumerate(agent_names):
        with cols[i]:
            st.markdown(f"**{name}**")
            st.metric("Inventory", f"{invs[i]:.1f}")
            st.metric("Backlog", f"{backs[i]:.1f}")
            st.metric("Current Order", f"{ords[i]:.1f}")
            st.markdown(f"<div style='background-color:{agent_status[i][1]}; color:{agent_status[i][2]}; padding:5px; border-radius:3px; text-align:center;'>{agent_status[i][0]}</div>", unsafe_allow_html=True)
            
    st.markdown("---")
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("What is happening right now?")
        if backs[3] > 0 and invs[3] < 5:
            st.write("🏭 **Manufacturer** is currently unable to satisfy downstream demand, stock is critically low.")
        elif backs[3] == 0 and t > 80 and t < 100:
            st.write("🏭 **Manufacturer** production is halted, but buffer inventory is currently absorbing the shock.")
            
        if backs[2] > 0:
            st.write("🚚 **Distributor** has run out of stock and is accumulating backlog.")
        elif invs[2] < 5 and t > 80:
            st.write("🚚 **Distributor** inventory is being rapidly depleted by the disruption shockwave.")
            
        if backs[1] > 0:
            st.write("📦 **Warehouse** shortages are occurring.")
            
        if backs[0] > 0:
            st.error("🏪 **Retailer** is experiencing customer-facing shortages! Service level is deteriorating rapidly.")
        elif invs[0] > 0 and backs[3] > 0:
            st.write("🏪 **Retailer** is still fulfilling customer demand using safety stock, shielding the end-consumer.")
        
        if sum(backs) == 0 and t < 80:
            st.success("The system is operating under normal steady-state conditions.")
            
    with colB:
        st.subheader("WHY DID THIS HAPPEN?")
        if backs[0] > 0:
            st.markdown("""
            **Retailer backlog increased because:**
            * Upstream inventory at the Warehouse/Distributor declined
            * Replenishment is physically delayed by the L=4 lead time
            * Customer demand continued to hit the Retailer
            """)
            st.warning("Current Bottleneck: **Manufacturer / Distributor**")
        elif backs[2] > 0:
            st.markdown("""
            **Distributor backlog increased because:**
            * Manufacturer ceased shipments.
            * Downstream orders from the Warehouse continued arriving.
            """)
            st.warning("Current Bottleneck: **Manufacturer**")
        else:
            st.info("No severe causal chains detected at this timestep. Supply chain is absorbing variance.")

    st.markdown("---")
    st.subheader("Historical Trajectories")
    
    ts_df = pd.DataFrame(demo1_data["timesteps"])
    
    # Restoring all historical charts for Demo 1
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_inv = go.Figure()
        for i, name in enumerate(agent_names):
            inv_vals_hist = [ts_['inventory'].get(name.lower(), ts_['inventory'].get(str(i), 0)) if isinstance(ts_['inventory'], dict) else ts_['inventory'][i] for ts_ in demo1_data["timesteps"]]
            fig_inv.add_trace(go.Scatter(x=ts_df["t"], y=inv_vals_hist, name=name, line=dict(color=colors[i])))
        fig_inv.add_vrect(x0=80, x1=99, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Disruption")
        fig_inv.add_vline(x=t, line_dash="solid", line_color="black", annotation_text="Current t")
        fig_inv.update_layout(title="Inventory Dynamics", height=300, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_inv, use_container_width=True)
        
    with col_chart2:
        fig_back = go.Figure()
        for i, name in enumerate(agent_names):
            back_vals_hist = [ts_['backlog'].get(name.lower(), ts_['backlog'].get(str(i), 0)) if isinstance(ts_['backlog'], dict) else ts_['backlog'][i] for ts_ in demo1_data["timesteps"]]
            fig_back.add_trace(go.Scatter(x=ts_df["t"], y=back_vals_hist, name=name, line=dict(color=colors[i])))
        fig_back.add_vrect(x0=80, x1=99, fillcolor="red", opacity=0.1)
        fig_back.add_vline(x=t, line_dash="solid", line_color="black", annotation_text="Current t")
        fig_back.update_layout(title="Backlog Dynamics", height=300, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_back, use_container_width=True)

    col_chart3, col_chart4 = st.columns(2)
    with col_chart3:
        fig_ord = go.Figure()
        for i, name in enumerate(agent_names):
            ord_vals = [ts_['order_qty'].get(name.lower(), ts_['order_qty'].get(str(i), 0)) if isinstance(ts_['order_qty'], dict) else ts_['order_qty'][i] for ts_ in demo1_data["timesteps"]]
            fig_ord.add_trace(go.Scatter(x=ts_df["t"], y=ord_vals, name=name, line=dict(color=colors[i])))
        fig_ord.add_vrect(x0=80, x1=99, fillcolor="red", opacity=0.1)
        fig_ord.add_vline(x=t, line_dash="solid", line_color="black")
        fig_ord.update_layout(title="Order Quantity", height=300, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_ord, use_container_width=True)
        
    with col_chart4:
        fig_sl = go.Figure()
        if "service_level_rolling" in ts_df.columns:
            fig_sl.add_trace(go.Scatter(x=ts_df["t"], y=ts_df["service_level_rolling"], name="Rolling SL", line=dict(color="orange")))
        fig_sl.add_vrect(x0=80, x1=99, fillcolor="red", opacity=0.1)
        fig_sl.add_vline(x=t, line_dash="solid", line_color="black")
        fig_sl.update_layout(title="Rolling Service Level", height=300, yaxis_range=[0, 1.1], margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_sl, use_container_width=True)

    # Automatically progress if playing
    if st.session_state.playing:
        if st.session_state.replay_t < max_t:
            time.sleep(0.5)
            st.session_state.replay_t += 1
            st.rerun()
        else:
            st.session_state.playing = False
            st.rerun()

# --- TAB 2: ALPHA IMPACT ---
with tab2:
    st.header("Block Replay Timeline")
    blocks = demo2_data["blocks"]
    n_blocks = len(blocks)
    
    if "replay_b" not in st.session_state:
        st.session_state.replay_b = 1
        
    # Clickable Timeline
    b_cols = st.columns(n_blocks)
    for i in range(n_blocks):
        b_num = i + 1
        with b_cols[i]:
            is_active = (st.session_state.replay_b == b_num)
            if st.button(f"Block {b_num}", type="primary" if is_active else "secondary", use_container_width=True):
                st.session_state.replay_b = b_num
                
    b_idx = st.session_state.replay_b - 1
    current_b = blocks[b_idx]
    
    st.subheader("CURRENT SYSTEM STATE")
    cols = st.columns(4)
    alphas = current_b["alpha_after_update"]
    kpis = current_b["block_kpis"]
    beh = kpis.get("agent_behavior_raw", kpis.get("agent_behavior", {}))
    
    for i, name in enumerate(agent_names):
        key = name.lower()
        with cols[i]:
            st.markdown(f"**{name}**")
            st.metric("Alpha Used", f"{alphas[key]:.2f}")
            st.metric("Cost Share", f"{kpis['cost_shares'][key]*100:.1f}%")
            if key in beh:
                st.metric("Avg Inventory", f"{beh[key]['avg_inventory']:.1f}")
                st.metric("Avg Backlog", f"{beh[key]['avg_backlog']:.1f}")
            
    st.markdown("---")
    
    st.subheader("WHY DID THE LLM CHANGE ALPHA?")
    st.info("LLM ANALYSIS PANEL")
    
    col_text, col_why = st.columns([2, 1])
    with col_text:
        st.markdown(f"**Reasoning:** {current_b['llm_output']['reasoning']}")
        st.markdown(f"**Self Correction:** {current_b['llm_output'].get('self_correction', 'N/A')}")
        st.markdown(f"**Rule Added:** {current_b['llm_output']['compiled_rule']}")
        
    with col_why:
        st.markdown("**Causal Breakdown:**")
        # Find the agent with highest cost share to highlight
        highest_agent = max(kpis['cost_shares'], key=kpis['cost_shares'].get)
        st.markdown(f"* Focused Agent: **{highest_agent.capitalize()}**")
        st.markdown(f"* Cost Share: **{kpis['cost_shares'][highest_agent]*100:.1f}%**")
        if highest_agent in beh:
            ratio = beh[highest_agent]['avg_inventory'] / max(1.0, beh[highest_agent]['avg_backlog'])
            st.markdown(f"* Inv/Backlog Ratio: **{ratio:.2f}**")
            b_type = "stock_starved" if ratio < 0.2 else "inventory_heavy" if ratio > 5 else "balanced"
            st.markdown(f"* Bottleneck: **{b_type}**")
            
        st.markdown(f"* Applied Alpha: **{alphas[highest_agent]:.2f}**")
        
    st.markdown("---")
    st.subheader("Historical Trajectories")
    
    summary2 = demo2_data["summary"]
    cost_traj = summary2["total_cost_trajectory"]
    sl_traj = summary2["service_level_trajectory"]
    bw_traj = summary2["bullwhip_trajectory"]
    
    x_vals = list(range(1, n_blocks+1))
    
    # Restoring all historical charts for Demo 2
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        fig_c = go.Figure(go.Bar(x=x_vals, y=cost_traj, marker_color=["#1f77b4" if i != b_idx else "#ff7f0e" for i in range(n_blocks)]))
        fig_c.update_layout(title="Total Cost", height=300, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_c, use_container_width=True)
    with col_kpi2:
        fig_s = go.Figure(go.Bar(x=x_vals, y=sl_traj, marker_color=["#1f77b4" if i != b_idx else "#ff7f0e" for i in range(n_blocks)]))
        fig_s.update_layout(title="Service Level", height=300, yaxis_range=[0, 1], margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_s, use_container_width=True)
    with col_kpi3:
        fig_b = go.Figure(go.Bar(x=x_vals, y=bw_traj, marker_color=["#1f77b4" if i != b_idx else "#ff7f0e" for i in range(n_blocks)]))
        fig_b.update_layout(title="Bullwhip Ratio", height=300, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_b, use_container_width=True)

    col_chartA, col_chartB = st.columns(2)
    with col_chartA:
        st.markdown("**Alpha Trajectories**")
        fig_alpha = go.Figure()
        for agent in agent_names:
            agent_key = agent.lower()
            agent_alphas = [b["alpha_after_update"][agent_key] for b in blocks]
            fig_alpha.add_trace(go.Scatter(x=x_vals, y=agent_alphas, name=agent, mode='lines+markers', marker=dict(size=10)))
        fig_alpha.add_vline(x=b_idx+1, line_dash="solid", line_color="black")
        fig_alpha.update_layout(height=350, yaxis_range=[0, 1.0], xaxis=dict(tickvals=x_vals), margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_alpha, use_container_width=True)

    with col_chartB:
        st.markdown("**Cost Share Distribution**")
        fig_share = go.Figure()
        for agent in agent_names:
            agent_key = agent.lower()
            shares = [b["block_kpis"]["cost_shares"][agent_key] * 100 for b in blocks]
            fig_share.add_trace(go.Bar(name=agent, x=x_vals, y=shares))
        fig_share.add_vline(x=b_idx+1, line_dash="solid", line_color="black")
        fig_share.update_layout(barmode='stack', height=350, yaxis_title="Percentage (%)", margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_share, use_container_width=True)
