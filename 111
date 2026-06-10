import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Spin Coating Simulator", layout="wide")

RHO = 1100.0

def omega_si(rpm):
    return rpm * 2.0 * np.pi / 60.0

def compute_tc(w, e0, h0):
    return 3.0 * e0*1e-3 / (RHO * omega_si(w)**2 * (h0*1e-6)**2)

def compute_Ne(w, e0, h0, E):
    return (E*1e-6) * compute_tc(w, e0, h0) / (h0*1e-6)

def h_gel_star(w, e0, h0, E):
    o = omega_si(w)
    h_dim = (3*e0*1e-3*(E*1e-6) / (2*RHO*o**2))**(1/3)
    return h_dim / (h0*1e-6)

def rhs(h, r_inner, r_half, dr, Ne):
    h_ext = np.concatenate([[h[0]], h, [h[-1]]])
    h_avg = 0.5 * (h_ext[:-1] + h_ext[1:])
    F = r_half**2 * h_avg**4
    div = (F[1:] - F[:-1]) / (r_inner * dr)
    return -div - Ne

def simulate(w, e0, h0, E, N=80, t_max=50.0):
    Ne = compute_Ne(w, e0, h0, E)
    tc = compute_tc(w, e0, h0)
    dr = 1.0 / N
    r_inner = np.linspace(dr/2, 1.0 - dr/2, N)
    r_half = np.linspace(0.0, 1.0, N+1)
    dt = 0.4 * dr / 4.0
    h = np.ones(N)
    t = 0.0
    hg = h_gel_star(w, e0, h0, E)
    t_gel = None
    ts = [0.0]
    hs = [h.copy()]
    while t < t_max:
        k1 = rhs(h, r_inner, r_half, dr, Ne)
        k2 = rhs(h + 0.5*dt*k1, r_inner, r_half, dr, Ne)
        k3 = rhs(h + 0.5*dt*k2, r_inner, r_half, dr, Ne)
        k4 = rhs(h + dt*k3, r_inner, r_half, dr, Ne)
        h = np.maximum(h + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4), 1e-4)
        t += dt
        if t_gel is None and h[0] <= hg:
            t_gel = t
            ts.append(t)
            hs.append(h.copy())
            break
        if t - ts[-1] >= 0.18:
            ts.append(t)
            hs.append(h.copy())
    return np.array(ts), np.array(hs), t_gel, r_inner, Ne, tc

def fig_style(fig, height=280):
    fig.update_layout(height=height, margin=dict(t=30,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(size=11))
    fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0")

st.title("Spin Coating Simulator")
tab1, tab2, tab3 = st.tabs(["Core Interactive", "Validation View", "Design Exploration"])

with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    w  = c1.slider("ω (rpm)", 500, 8000, 3000, 100)
    e0 = c2.slider("η₀ (mPa·s)", 1.0, 100.0, 10.0, 0.5)
    h0 = c3.slider("h₀ (μm)", 2.0, 50.0, 15.0, 0.5)
    E  = c4.slider("E (μm/s)", 0.01, 1.0, 0.10, 0.01)
    wr = c5.slider("Wafer radius (mm)", 50, 150, 100, 10)

    with st.spinner("running..."):
        ts, hs, tg, r_inner, Ne, tc = simulate(w, e0, h0, E)

    hg = h_gel_star(w, e0, h0, E)
    tg_s = tg * tc if tg else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ne", f"{Ne:.4f}")
    m2.metric("tc (s)", f"{tc:.3f}")
    m3.metric("h*_gel", f"{hg:.4f}")
    m4.metric("t_gel (s)", f"{tg_s:.2f}" if tg_s else "—")

    r_mm = r_inner * wr
    r_full = np.concatenate([-r_mm[::-1], r_mm])
    n_frames = min(len(ts), 40)
    frame_idx = np.linspace(0, len(ts)-1, n_frames, dtype=int)

    frames = []
    for fi in frame_idx:
        h_dim = hs[fi, :] * h0
        y_full = np.concatenate([h_dim[::-1], h_dim])
        frames.append(go.Frame(
            data=[go.Scatter(x=r_full, y=y_full, fill="tozeroy",
                fillcolor="rgba(0,0,0,0.08)", line=dict(color="#111", width=2))],
            name=str(fi),
            layout=go.Layout(title_text=f"t = {ts[fi]*tc:.2f} s  (t* = {ts[fi]:.2f})")
        ))

    h0_full = np.concatenate([hs[0,::-1], hs[0,:]]) * h0
    fig_anim = go.Figure(
        data=[go.Scatter(x=r_full, y=h0_full, fill="tozeroy",
            fillcolor="rgba(0,0,0,0.08)", line=dict(color="#111", width=2))],
        frames=frames,
        layout=go.Layout(
            title="t = 0.00 s  (t* = 0.00)",
            xaxis_title="r (mm)", yaxis_title="h (μm)",
            yaxis=dict(range=[0, h0 * 1.1]),
            height=380, margin=dict(t=50, b=40, l=50, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(size=11),
            updatemenus=[dict(type="buttons", showactive=False, y=1.15, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                        args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ Pause", method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ])],
            sliders=[dict(
                steps=[dict(method="animate", args=[[str(fi)],
                    dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                    label=f"{ts[fi]*tc:.1f}s") for fi in frame_idx],
                x=0, y=-0.05, len=1.0,
                currentvalue=dict(prefix="t = ", suffix=" s", visible=True, xanchor="center"),
            )]
        )
    )
    fig_anim.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
    fig_anim.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
    st.plotly_chart(fig_anim, use_container_width=True)

    N_nodes = len(r_inner)
    idxs  = [0, N_nodes//4, N_nodes//2, 3*N_nodes//4, N_nodes-1]
    lbls  = ["r*=0", f"r*={r_inner[N_nodes//4]:.2f}", f"r*={r_inner[N_nodes//2]:.2f}",
             f"r*={r_inner[3*N_nodes//4]:.2f}", "r*=1"]
    dashes = ["solid","dot","dash","dashdot","longdash"]
    grays  = ["#111","#444","#777","#aaa","#ccc"]

    fig_ht = go.Figure()
    for idx, lbl, ds, gc in zip(idxs, lbls, dashes, grays):
        fig_ht.add_trace(go.Scatter(x=ts*tc, y=hs[:,idx]*h0, name=lbl,
            line=dict(color=gc, width=1.8, dash=ds)))
    if tg_s:
        fig_ht.add_vline(x=tg_s, line_dash="dot", line_color="#aaa", line_width=1,
            annotation_text=f"t_gel={tg_s:.2f}s", annotation_font_size=9)
    fig_ht.update_layout(xaxis_title="t (s)", yaxis_title="h (μm)",
        legend=dict(orientation="h", x=0, y=-0.2, font_size=10))
    fig_style(fig_ht, height=300)
    st.plotly_chart(fig_ht, use_container_width=True)

with tab2:
    st.markdown("**① Limit Check — Ne → 0**")
    st.markdown("pure centrifugal limit: no evaporation → analytical solution from EBP (1958)")
    st.latex(r"h^*(t^*) = \left(1 + 6t^*\right)^{-1/3}")

    ts_l, hs_l, _, _, Ne_l, _ = simulate(w, e0, h0, 1e-4, t_max=8.0)
    h_ebp = (1.0 + 6.0 * ts_l)**(-1/3)
    err = np.abs(hs_l[:, 0] - h_ebp)

    fig_v1 = make_subplots(rows=1, cols=2,
        subplot_titles=["h*(t*): numerical vs analytical", "|error|"],
        horizontal_spacing=0.12)
    fig_v1.add_trace(go.Scatter(x=ts_l, y=hs_l[:,0], name="numerical",
        line=dict(color="#111", width=2)), row=1, col=1)
    fig_v1.add_trace(go.Scatter(x=ts_l, y=h_ebp, name="(1+6t*)^(-1/3)",
        line=dict(color="#aaa", width=1.5, dash="dot")), row=1, col=1)
    fig_v1.add_trace(go.Scatter(x=ts_l, y=err, name="|error|",
        line=dict(color="#555", width=1.5)), row=1, col=2)
    fig_v1.update_xaxes(title_text="t*", row=1, col=1)
    fig_v1.update_yaxes(title_text="h*", row=1, col=1)
    fig_v1.update_xaxes(title_text="t*", row=1, col=2)
    fig_v1.update_yaxes(title_text="|error|", row=1, col=2)
    fig_style(fig_v1, height=300)
    fig_v1.update_layout(legend=dict(orientation="h", x=0, y=-0.2, font_size=10))
    st.plotly_chart(fig_v1, use_container_width=True)
    l2 = float(np.sqrt(np.mean(err**2)))
    st.caption(f"Ne = {Ne_l:.2e}  |  L2 error = {l2:.2e}")

    st.markdown("---")
    st.markdown("**② Parametric Trend — h_final ∝ ω^(-2/3)**")
    st.markdown("Meyerhofer (1978):  h_f ∝ ω^(-2/3) η₀^(1/3) E^(1/3)")

    with st.spinner("scanning ω..."):
        omega_range = np.arange(500, 7100, 300)
        hf_num = []
        for ww in omega_range:
            _, hs_w, _, _, _, _ = simulate(ww, e0, h0, E, N=60, t_max=50.0)
            hf_num.append(hs_w[-1, 0] * h0)

    hf_num = np.array(hf_num)
    hf_mey = (3*e0*1e-3/(2*RHO))**(1/3) * (E*1e-6)**(1/3) / omega_si(omega_range)**(2/3) * 1e6

    log_w = np.log10(omega_range)
    log_h = np.log10(hf_num)
    slope = np.polyfit(log_w, log_h, 1)

    fig_v2 = make_subplots(rows=1, cols=2,
        subplot_titles=["h_final vs ω", "log-log (slope check)"],
        horizontal_spacing=0.12)
    fig_v2.add_trace(go.Scatter(x=omega_range, y=hf_num, name="numerical",
        line=dict(color="#111", width=2)), row=1, col=1)
    fig_v2.add_trace(go.Scatter(x=omega_range, y=hf_mey, name="Meyerhofer analytical",
        line=dict(color="#aaa", width=1.5, dash="dot")), row=1, col=1)
    fig_v2.add_vline(x=w, line_dash="dot", line_color="#ccc", line_width=1, row=1, col=1)
    fig_v2.add_trace(go.Scatter(x=log_w, y=log_h, mode="markers",
        marker=dict(color="#111", size=5), name="numerical"), row=1, col=2)
    fig_v2.add_trace(go.Scatter(x=log_w, y=np.polyval(slope, log_w),
        name=f"fit: slope={slope[0]:.3f}", line=dict(color="#aaa", dash="dot")), row=1, col=2)
    fig_v2.update_xaxes(title_text="ω (rpm)", row=1, col=1)
    fig_v2.update_yaxes(title_text="h_final (μm)", row=1, col=1)
    fig_v2.update_xaxes(title_text="log10(ω)", row=1, col=2)
    fig_v2.update_yaxes(title_text="log10(h_final)", row=1, col=2)
    fig_style(fig_v2, height=320)
    fig_v2.update_layout(legend=dict(orientation="h", x=0, y=-0.2, font_size=10))
    st.plotly_chart(fig_v2, use_container_width=True)
    st.caption(f"fitted slope: {slope[0]:.3f}  (theory: -2/3 = -0.667)")

with tab3:
    st.markdown("Find (ω, η₀) combinations that meet target h_final")
    col_l, col_r = st.columns([1, 2])

    with col_l:
        tol   = st.number_input("h_final tolerance (±%)", 0.5, 20.0, 2.0, 0.5)
        t_h   = st.number_input("Target h_final (μm)", 0.1, 20.0, 2.0, 0.1)
        E_de  = st.slider("E (μm/s)", 0.01, 1.0, float(E), 0.01, key="E_de")
        h0_de = st.slider("h₀ (μm)", 2.0, 50.0, float(h0), 0.5, key="h0_de")
        wmin  = st.slider("ω min (rpm)", 500, 3000, 1000, 100)
        wmax  = st.slider("ω max (rpm)", 3000, 8000, 6000, 100)
        emin  = st.slider("η₀ min (mPa·s)", 1.0, 30.0, 2.0, 1.0)
        emax  = st.slider("η₀ max (mPa·s)", 10.0, 100.0, 50.0, 1.0)
        n_pts = st.select_slider("Grid resolution", [8, 10, 12, 15], value=10)
        go_btn = st.button("Run scan", type="primary", use_container_width=True)

    with col_r:
        if go_btn:
            wa = np.linspace(wmin, wmax, n_pts)
            ea = np.linspace(emin, emax, n_pts)
            hf_map = np.zeros((n_pts, n_pts))
            tg_map = np.zeros((n_pts, n_pts))
            Ne_map = np.zeros((n_pts, n_pts))

            prog = st.progress(0, text="scanning...")
            for i, wi in enumerate(wa):
                for j, ei in enumerate(ea):
                    _, hs_s, tg_s, _, Ne_s, tc_s = simulate(wi, ei, h0_de, E_de, N=60, t_max=50.0)
                    hf_map[j,i] = hs_s[-1, 0] * h0_de
                    tg_map[j,i] = tg_s * tc_s if tg_s else 0.0
                    Ne_map[j,i] = Ne_s
                prog.progress((i+1)/n_pts, text=f"ω={wi:.0f} rpm")
            prog.empty()

            gs = [[0.0,"#111"],[0.5,"#888"],[1.0,"#eee"]]
            fig_de = make_subplots(rows=1, cols=2,
                subplot_titles=["h_final (μm)", "t_gel (s)"],
                horizontal_spacing=0.10)
            fig_de.add_trace(go.Heatmap(x=wa, y=ea, z=hf_map,
                colorscale=gs, colorbar=dict(title="μm", x=0.45, len=0.9)), row=1, col=1)
            fig_de.add_trace(go.Contour(x=wa, y=ea, z=hf_map,
                contours=dict(start=t_h*(1-tol/100), end=t_h*(1+tol/100), size=t_h*tol/50,
                    coloring="none", showlabels=True, labelfont=dict(size=10, color="white")),
                line=dict(color="white", width=2, dash="dash"), showscale=False), row=1, col=1)
            fig_de.add_trace(go.Heatmap(x=wa, y=ea, z=tg_map,
                colorscale=gs, colorbar=dict(title="s", x=1.0, len=0.9)), row=1, col=2)
            for c_ in [1, 2]:
                fig_de.update_xaxes(title_text="ω (rpm)", row=1, col=c_)
                fig_de.update_yaxes(title_text="η₀ (mPa·s)", row=1, col=c_)
            fig_de.update_layout(height=380, margin=dict(t=40,b=10,l=10,r=80),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False, font=dict(size=11))
            st.plotly_chart(fig_de, use_container_width=True)

            rows = []
            for i, wi in enumerate(wa):
                for j, ei in enumerate(ea):
                    if abs(hf_map[j,i] - t_h) / t_h * 100 <= tol:
                        rows.append({
                            "ω (rpm)": f"{wi:.0f}",
                            "η₀ (mPa·s)": f"{ei:.1f}",
                            "h_final (μm)": f"{hf_map[j,i]:.3f}",
                            "t_gel (s)": f"{tg_map[j,i]:.2f}",
                            "Ne": f"{Ne_map[j,i]:.4f}",
                        })
            if rows:
                st.success(f"{len(rows)} combinations within ±{tol}% of target {t_h} μm")
                st.dataframe(rows, use_container_width=True, height=200)
            else:
                st.warning("no combinations meet the target — try adjusting ranges")
        else:
            st.info("set ranges and click Run scan")
