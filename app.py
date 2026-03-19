import numpy as np
import streamlit as st
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plate_ode import nondim_freely_falling_plate

def apply_font(fig, size=16, family="Times New Roman, Times, serif"):
    fig.update_layout(
        font=dict(family=family, size=size, color="black"),
        title_font=dict(family=family, size=size, color="black"),
        legend=dict(font=dict(family=family, size=size, color="black")),
    )
    fig.update_xaxes(
        tickfont=dict(family=family, size=size, color="black"),
        title_font=dict(family=family, size=size, color="black"),
    )
    fig.update_yaxes(
        tickfont=dict(family=family, size=size, color="black"),
        title_font=dict(family=family, size=size, color="black"),
    )
    # for 3D plots
    for axis in ("xaxis", "yaxis", "zaxis"):
        fig.update_layout({
            f"scene_{axis}_tickfont": dict(family=family, size=size, color="black"),
            f"scene_{axis}_title_font": dict(family=family, size=size, color="black"),
        })
    return fig

def parse_coeffs(s: str, expected_n: int):
    # Accept commas or whitespace
    parts = [p for p in s.replace(",", " ").split() if p.strip()]
    vals = [float(p) for p in parts]
    if len(vals) != expected_n:
        raise ValueError(f"Expected {expected_n} numbers, got {len(vals)}")
    return np.array(vals, dtype=float)


def downsample_for_plot(x, y, max_points=20000):
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return x[idx], y[idx]


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="2D falling plate ODE", layout="wide")
st.title("Two-dimensional freely-falling plate ODE (Li *et al.*)")
st.write("Code hosted at https://github.com/ckessler2/Plate_Model_Site")

st.markdown(
    """
    <style>
      :root {
        --app-font: "Times New Roman", Times, serif;
      }

      /* Headings (st.title / header / subheader and Markdown headings) */
      h1, h2, h3, h4, h5, h6 {
        font-family: var(--app-font) !important;
      }

      /* Markdown blocks explicitly (covers some edge cases) */
      div[data-testid="stMarkdownContainer"] * {
        font-family: var(--app-font) !important;
      }
      
      /* Hide the sidebar collapse/expand arrow buttons (top-left arrows) */
        button[data-testid="stSidebarCollapseButton"],
        button[data-testid="stSidebarExpandButton"]{
          display: none !important;
        }
        
        /* Hide number_input spinner buttons (prevents arrow_up/arrow_down text) */
        button[aria-label="Increment"],
        button[aria-label="Decrement"]{
          display: none !important;
        }
        /* Pin the Run ODE button to the top of the sidebar */
        section[data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            width: 600px !important;
            min-width: 500px !important;
        }
        
        div[data-testid="stSidebarUserContent"] {
            display: flex;
            flex-direction: column;
        }
        
        /* Target the first button in the sidebar and stick it */
        section[data-testid="stSidebar"] button[kind="primary"] {
            position: sticky;
            top: 0.5rem;
            z-index: 999;
            width: 100%;
        }
        .block-container {
       padding-top: 2rem;

      
    </style>
    """,
    unsafe_allow_html=True,
)



with st.sidebar:
    top = st.columns([0.32, 0.38])        # or put a small title/blank spacer
    run = top[1].button("Run ODE and plot", type="primary", use_container_width=True)
    top[0].header("Constants")
    c = st.columns(6)
    c[0].markdown("<i>l</i>", unsafe_allow_html=True);     l     = c[0].number_input("l",     value=0.07,     format="%.8g", key="l",     label_visibility="collapsed")
    c[1].markdown("<i>m</i>", unsafe_allow_html=True);     m     = c[1].number_input("m",     value=0.36e-3,  format="%.8g", key="m",     label_visibility="collapsed")
    c[2].markdown("ρ<sub>f</sub>", unsafe_allow_html=True); rho_f = c[2].number_input("rho_f", value=1.225,   format="%.8g", key="rho_f", label_visibility="collapsed")
    c[3].markdown("<i>a</i>", unsafe_allow_html=True);     a     = c[3].number_input("a",     value=0.03375,  format="%.8g", key="a",     label_visibility="collapsed")
    c[4].markdown("<i>b</i>", unsafe_allow_html=True);     b     = c[4].number_input("b",     value=0.5e-3,   format="%.8g", key="b",     label_visibility="collapsed")
    c[5].markdown("<i>s</i>", unsafe_allow_html=True);     s     = c[5].number_input("s",     value=0.1745,   format="%.8g", key="s",     label_visibility="collapsed")
    
    st.header("Initial conditions")
    ic = st.columns(6)
    ic[0].markdown("<i>v</i><sub>x′</sub>(0)", unsafe_allow_html=True); v_xp0  = ic[0].number_input("v_xp0",  value=0.0,         format="%.8g", key="v_xp0",  label_visibility="collapsed")
    ic[1].markdown("<i>v</i><sub>y′</sub>(0)", unsafe_allow_html=True); v_yp0  = ic[1].number_input("v_yp0",  value=0.0,         format="%.8g", key="v_yp0",  label_visibility="collapsed")
    ic[2].markdown("ω(0)", unsafe_allow_html=True);       omega0 = ic[2].number_input("omega0", value=0.0,         format="%.8g", key="omega0", label_visibility="collapsed")
    ic[3].markdown("θ(0)", unsafe_allow_html=True);       theta0 = ic[3].number_input("theta0", value=-1.0,  format="%.8g", key="theta0", label_visibility="collapsed")
    ic[4].markdown("<i>x</i>(0)", unsafe_allow_html=True);             x0     = ic[4].number_input("x0",     value=0.0,         format="%.8g", key="x0",     label_visibility="collapsed")
    ic[5].markdown("<i>y</i>(0)", unsafe_allow_html=True);             y0     = ic[5].number_input("y0",     value=0.0,         format="%.8g", key="y0",     label_visibility="collapsed")
    

    st.header("Aerodynamic coefficients")
    default_coeffs = "5.182 0.8075 0.1060 4.937 1.500 0.2386 2.853 0.3689 1.730"
    st.markdown(
        "<i>C</i><sub>L1</sub>  <i>C</i><sub>L2</sub>  "
        "<i>C</i><sub>D0</sub>  <i>C</i><sub>D1</sub>  "
        "<i>C</i><sub>D,π/2</sub>  "
        "<i>C</i><sub>0,CP</sub>  <i>C</i><sub>1,CP</sub>  <i>C</i><sub>2,CP</sub>  "
        "<i>C</i><sub>R</sub>",
        unsafe_allow_html=True,
    )
    coeffs_str = st.text_input(
        "aero_coeffs",
        value=default_coeffs,
        label_visibility="collapsed",
        key="aero_coeffs",
    )

    st.header("Chordwise CoM eccentricity")
    ex_c = st.columns(2)
    ex_c[0].markdown("<i>e</i><sub><i>x</i></sub>",unsafe_allow_html=True)
    e_x = ex_c[1].number_input("<i>e</i><sub><i>x</i></sub>",label_visibility="collapsed", value=0.0, format="%.8g")

    st.header("Solver and plotting settings")
    tg = st.columns(5)
    
    tg[0].markdown("timestep", unsafe_allow_html=True)
    dt = tg[0].number_input(
        "dt", value=0.05, min_value=1e-6, format="%.8g",
        key="dt", label_visibility="collapsed"
    )
    
    tg[1].markdown("final time", unsafe_allow_html=True)
    t_end = tg[1].number_input(
        "t_end", value=25.0, min_value=1e-6, format="%.8g",
        key="t_end", label_visibility="collapsed"
    )

    # sp = st.columns(3)
    
    tg[2].markdown("rtol", unsafe_allow_html=True)
    rtol = tg[2].number_input(
        "rtol", value=1e-6, format="%.2e",
        key="rtol", label_visibility="collapsed"
    )
    
    tg[3].markdown("atol", unsafe_allow_html=True)
    atol = tg[3].number_input(
        "atol", value=1e-9, format="%.2e",
        key="atol", label_visibility="collapsed"
    )
    
    tg[4].markdown("max points", unsafe_allow_html=True)
    max_plot_points = tg[4].number_input(
        "max_plot_points", value=20000, min_value=1000, step=1000,
        key="max_plot_points", label_visibility="collapsed"
    )
    
    sp2 = st.columns(2)
    sp2[0].markdown("Orthographic projection", unsafe_allow_html=True)
    proj = "orthographic" if sp2[0].toggle("", value=True) else "perspective"
    n_plates = sp2[1].number_input(
        "3d scene plate count", value=40, min_value=1, step=1,
    )
    


if run:
    try:
        aero9 = parse_coeffs(coeffs_str, expected_n=9)
    except Exception as ex:
        st.error(f"Could not parse coefficients: {ex}")
        st.stop()

    opt = np.concatenate([aero9, [float(e_x)]])
    p = np.array([l, m, rho_f, a, b, s], dtype=float)
    y_init = np.array([v_xp0, v_yp0, omega0, theta0, x0, y0], dtype=float)

    # Exact MATLAB-style output times:
    # t = 0:dt:t_end (inclusive end if it lands on grid)
    n_steps = int(np.floor(t_end / dt))
    t_eval = np.arange(0.0, (n_steps + 1) * dt, dt, dtype=float)

    # st.write(f"Evaluating for **{len(t_eval):,}** time points.")
    

    with st.spinner("Integrating ..."):
        sol = solve_ivp(
            fun=lambda t, y: nondim_freely_falling_plate(t, y, p, opt),
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=y_init,
            t_eval=t_eval,
            method="RK45",           # closest to MATLAB ode45
            rtol=float(rtol),
            atol=float(atol),
            max_step=float(dt),      # encourage similar stepping behaviour to MATLAB output grid
        )

    if not sol.success:
        st.error(f"Solver failed: {sol.message}")
        st.stop()

    v_xp = sol.y[0]
    v_yp = sol.y[1]
    omega = sol.y[2]
    theta = sol.y[3]
    x_pos = sol.y[4]
    y_pos = sol.y[5]
    
    # --- axis limits to nearest multiples of 10 ---
    base = 10.0
    xmin = base * np.floor(np.nanmin(x_pos) / base)
    xmax = base * np.ceil (np.nanmax(x_pos) / base)
    ymin = base * np.floor(np.nanmin(y_pos) / base)
    ymax = base * np.ceil (np.nanmax(y_pos) / base)
    
    # handle degenerate ranges (all values identical)
    if xmin == xmax:
        xmin -= base
        xmax += base
    if ymin == ymax:
        ymin -= base
        ymax += base
    
    tgp = st.columns(2)
    with tgp[0]:

        st.subheader("*x* versus *y*")
        x_plot, y_plot = downsample_for_plot(x_pos, y_pos, max_points=int(max_plot_points))
    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode="lines", name="<i>x</i>(t) vs <i>y</i>(t)",cliponaxis=True))
        fig.add_trace(go.Scatter(x=[x_pos[0]], y=[y_pos[0]], mode="markers", name="start",cliponaxis=True))
        fig.add_trace(go.Scatter(x=[x_pos[-1]], y=[y_pos[-1]], mode="markers", name="end",cliponaxis=True))
        fig.update_layout(
            xaxis_title="</i>x</i>",
            yaxis_title="</i>y</i>",
            height=600,title=dict(text=""), 
            margin=dict(l=40, r=20, t=40, b=40),
            font=dict(family="Times New Roman, Times, serif",color="black"),
            paper_bgcolor="white",
            plot_bgcolor="white",       # all text (ticks/legend/title) black
            title_font=dict(color="black"),
            legend=dict(font=dict(color="black"),x=1,
                y=1,
                xanchor="right",
                yanchor="top"
            ),
            )
        
        fig.update_xaxes(range=[xmin, xmax], dtick=10, autorange=False, constrain="domain", fixedrange=True,
            showgrid=False, zeroline=False,
            showline=True, linecolor="black", linewidth=1, mirror=True,
            tickfont=dict(color="black"),
            title_font=dict(color="black"),
            tickcolor="black",
        )
        fig.update_yaxes(range=[ymin, ymax], dtick=10, autorange=False, constrain="domain", fixedrange=True,
            showgrid=False, zeroline=False,
            showline=True, linecolor="black", linewidth=1, mirror=True,
            tickfont=dict(color="black"),
            title_font=dict(color="black"),
            tickcolor="black",
            scaleanchor="x", scaleratio=1,
        )
        
        fig.update_xaxes(layer="above traces")
        fig.update_yaxes(layer="above traces")

    
        st.plotly_chart(apply_font(fig))
    
    
    with tgp[1]:
        from plate_3d import plate_scene_xz

    
        L_plate = float(1)         # as you requested
        W_plate = float(2)   # example width; change if needed
        
        
    
        fig3d = plate_scene_xz(
            t=sol.t,
            x=x_pos,
            z=y_pos,
            theta=theta,
            n=n_plates,
            L=L_plate,
            W=W_plate,
            camera_eye=(1.8, 1.2, 0.6),
            projection=proj,
        )
        fig3d.update_layout(
            xaxis_title="</i>x</i>",
            yaxis_title="</i>y</i>",
            height=600,title=dict(text=""), 
            margin=dict(l=40, r=20, t=40, b=40),
            font=dict(family="Times New Roman, Times, serif",color="black"),
            paper_bgcolor="white",
            plot_bgcolor="white",       # all text (ticks/legend/title) black
            title_font=dict(color="black"),
            legend=dict(font=dict(color="black"),x=1,
                y=1,
                xanchor="right",
                yanchor="top"
            ),
            )
        st.subheader("3D plot")
        st.plotly_chart(apply_font(fig3d), key="plates_scene_3d")

    st.subheader("Parameters versus time")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sol.t, y=v_xp,  mode="lines", name="<i>v</i><sub>x′</sub>"))
    fig2.add_trace(go.Scatter(x=sol.t, y=v_yp,  mode="lines", name="<i>v</i><sub>y′</sub>"))
    fig2.add_trace(go.Scatter(x=sol.t, y=omega, mode="lines", name="ω"))
    fig2.add_trace(go.Scatter(x=sol.t, y=theta, mode="lines", name="θ"))
    fig2.update_layout(xaxis_title="<i>t</i>", yaxis_title="<i>v</i><sub>x′</sub>, <i>v</i><sub>y′</sub>, ω, θ", height=400,
    font=dict(family="Times New Roman, Times, serif",color="black"),
    paper_bgcolor="white",title=dict(text=""), 
    plot_bgcolor="white",
    title_font=dict(color="black"),
    legend=dict(font=dict(color="black")),
    
    )
    fig2.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor="black", linewidth=1, mirror=True)
    fig2.update_yaxes(showgrid=False, zeroline=False, showline=True, linecolor="black", linewidth=1, mirror=True)
    fig2.update_xaxes( dtick=10, constrain="domain", 
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        tickcolor="black",
        hoverformat=".4f",
    )
    fig2.update_yaxes( dtick=1, constrain="domain", 
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        tickcolor="black",tickmode="linear",
        tick0=0,
        tickformat=".0f",   # show as integers
        hoverformat=".4f",
        showticklabels=True,
        ticks="outside",
    )
    
    

    st.plotly_chart(apply_font(fig2), use_container_width=True)


    with st.expander("Final state", expanded=False):
        st.write(
            {
                "t_end": float(sol.t[-1]),
                "v_xp": float(v_xp[-1]),
                "v_yp": float(v_yp[-1]),
                "omega": float(omega[-1]),
                "theta": float(theta[-1]),
                "x": float(x_pos[-1]),
                "y": float(y_pos[-1]),
            }
        )
    
    with st.expander("ODE source", expanded=False):
        with open("plate_ode.py", "r") as f:
            st.code(f.read(), language="python")
    with st.expander("Page source", expanded=False):
        with open("app.py", "r", encoding="utf-8") as f:
            st.code(f.read(), language="python")
    
else:
    st.info("Set parameters in the sidebar, then click **Run ODE and plot**.")