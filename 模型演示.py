"""Reproducible conditional examples, with no API calls and no fitted data.

Run with the bundled Python runtime. Pillow renders the PNG; the SVG and all
numerical calculations use the Python standard library. These are two separate
modules, not a calibrated model of the Internet or a solved coupled equilibrium.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = ROOT / "outputs" / "模型演示结果.json"
FIGURE_DIR = ROOT / "outputs" / "figures"
R, TAU, FIXED, COST = 10000.0, 0.2, 2.0, 1.0
W = (1.0 - TAU) * R
INFO = {"S": 1.0, "alpha": 0.3, "theta": 0.4,
        "delta_H": 0.1, "beta": 0.2, "gamma": 0.1,
        "delta_B": 0.3, "eta": 0.3, "H0": 10.0, "B0": 1.0}
SCENARIOS = [(1.0, 0.2), (4.0, 0.2), (6.0, 0.2), (6.0, 0.6)]


def net_profit(n: int | float) -> float:
    return W / n**2 - FIXED


def golden_maximum(fn, low: float, high: float) -> tuple[float, float]:
    """Numerically optimize actual unilateral payoff, not equilibrium formula."""
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    for _ in range(160):
        if fn(left) > fn(right):
            high, right = right, left
            left = high - ratio * (high - low)
        else:
            low, left = left, right
            right = low + ratio * (high - low)
    x = (low + high) / 2.0
    return x, fn(x)


def contest_row(n: int) -> dict:
    effort = W * (n - 1.0) / (COST * n**2)
    rivals = (n - 1.0) * effort

    def unilateral_payoff(x):
        return W * x / (x + rivals) - COST * x - FIXED

    best_effort, best_profit = golden_maximum(unilateral_payoff, 0.0, W / COST)
    implied_profit = unilateral_payoff(effort)
    assert abs(implied_profit - net_profit(n)) < 1e-9
    assert abs(best_effort - effort) / effort < 1e-5
    assert abs(best_profit - implied_profit) < 1e-8
    assert unilateral_payoff(0.9 * effort) <= implied_profit + 1e-9
    assert unilateral_payoff(1.1 * effort) <= implied_profit + 1e-9
    total_effort = n * effort
    # Prize accounting: owners' profits + effort charges + fixed charges = W.
    assert abs(n * implied_profit + COST * total_effort + n * FIXED - W) < 1e-8
    return {"N": n, "per_agent_effort": effort,
            "total_effort": total_effort, "per_agent_net_profit": implied_profit,
            "total_owner_net_profit": n * implied_profit,
            "numerical_best_response_effort": best_effort,
            "best_response_effort_relative_error": abs(best_effort-effort)/effort}


def information_state(t: float, activity: float, verification: float) -> dict:
    p = INFO
    h_star = (p["S"] + p["alpha"] * p["theta"] * activity) / p["delta_H"]
    h = h_star + (p["H0"] - h_star) * math.exp(-p["delta_H"] * t)
    source = p["beta"] * (1.0 - p["theta"]) * activity
    r = p["gamma"] * activity * (1.0 - verification) - p["delta_B"] - p["eta"] * verification
    # expm1 avoids cancellation near r=0; the exact zero limit is linear.
    b = (p["B0"] + source * t) if abs(r) < 1e-14 else (
        p["B0"] * math.exp(r * t) + source * math.expm1(r * t) / r)
    return {"t": t, "H": h, "B": b, "q": h / (h + b)}


def rk4_check(activity: float, verification: float, end: float = 50.0) -> dict:
    """Independent ODE integration: RHS uses the differential equations."""
    p = INFO

    def rhs(h, b):
        return (p["S"] + p["alpha"] * p["theta"] * activity - p["delta_H"] * h,
                p["beta"] * (1.0 - p["theta"]) * activity
                + p["gamma"] * activity * (1.0 - verification) * b
                - p["delta_B"] * b - p["eta"] * verification * b)

    h, b, step = p["H0"], p["B0"], 0.01
    for _ in range(round(end / step)):
        k1 = rhs(h, b)
        k2 = rhs(h + step*k1[0]/2, b + step*k1[1]/2)
        k3 = rhs(h + step*k2[0]/2, b + step*k2[1]/2)
        k4 = rhs(h + step*k3[0], b + step*k3[1])
        h += step * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
        b += step * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
    exact = information_state(end, activity, verification)
    h_error = abs(h-exact["H"])/max(1.0, exact["H"])
    b_error = abs(b-exact["B"])/max(1.0, exact["B"])
    assert max(h_error, b_error) < 1e-8
    return {"method": "RK4 on the ODE RHS", "step": step,
            "H_relative_error": h_error, "B_relative_error": b_error}


def closed_feedback_counterexample() -> dict:
    """A one-dimensional extension, not a full economy/information equilibrium."""
    a0, kappa, b, g, d, h0 = 6.0, 0.5, 0.012, 0.08, 0.36, 10.0

    def activity(x):
        return a0 / (1.0 + kappa * x)

    def rhs(x):
        a = activity(x)
        return b * a + (g * a - d) * x

    x_star = (g*a0-d+math.sqrt((d-g*a0)**2+4*d*kappa*b*a0))/(2*d*kappa)
    low, high = 0.0, 1.0
    while rhs(high)>0:
        high *= 2
    for _ in range(100):
        midpoint = (low+high)/2
        if rhs(midpoint)>0:
            low = midpoint
        else:
            high = midpoint
    bisection_root = (low+high)/2
    a_prime = -a0*kappa/(1+kappa*x_star)**2
    derivative = b*a_prime+g*(activity(x_star)+x_star*a_prime)-d
    finite_difference = (rhs(x_star+1e-5)-rhs(x_star-1e-5))/2e-5
    assert abs(rhs(x_star))<1e-12
    assert abs(bisection_root-x_star)<1e-12
    assert derivative<0
    assert abs(derivative-finite_difference)<1e-9
    # Time integration starts away from the root and converges without using it.
    x, step, end = 0.1, 0.02, 500.0
    for _ in range(round(end/step)):
        k1 = rhs(x)
        k2 = rhs(x+step*k1/2)
        k3 = rhs(x+step*k2/2)
        k4 = rhs(x+step*k3)
        x += step*(k1+2*k2+2*k3+k4)/6
    assert abs(x-x_star)<1e-9
    return {"scope": "Single-state counterexample with frozen H=10 and activity reduced by pollution-related cost. It is a deliberately assumed feedback, not measured behavior, not a full coupled equilibrium.",
            "parameters": {"H0":h0,"A0":a0,"kappa":kappa,"b":b,"g":g,"d":d,"x_initial":0.1},
            "unit_convention": "x=B/H0 and q=1/(1+x) are dimensionless; H0 frozen; time uses the abstract units of the information module; activity declines as A0/(1+kappa*x).",
            "equations": {"A": "A0/(1+kappa*x)","x_dot":"b*A+(g*A-d)*x",
                          "positive_equilibrium":"[g*A0-d+sqrt((d-g*A0)^2+4*d*kappa*b*A0)]/(2*d*kappa)"},
            "x_equilibrium":x_star,"B_equilibrium":h0*x_star,
            "q_equilibrium":1/(1+x_star),"A_equilibrium":activity(x_star),
            "fixed_A0_amplification_rate":g*a0-d,
            "feedback_local_derivative":derivative,"locally_stable":derivative<0,
            "validation":{"bisection_root":bisection_root,"rk4_t_end":end,"rk4_dt":step,"rk4_x_end":x,
                          "rk4_absolute_equilibrium_error":abs(x-x_star),
                          "finite_difference_derivative":finite_difference},
            "interpretation": "Fixed A=6 has a positive noise-amplification rate. Endogenous activity reduction here yields a finite, stable polluted state, disproving an unconditional divergence claim within this extension."}


class Canvas:
    """Use the same plot geometry for portable SVG and high-resolution PNG."""

    def __init__(self, width=1700, height=950):
        self.width, self.height = width, height
        self.image = Image.new("RGB", (width, height), "#f5f7fb")
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    f'<rect width="{width}" height="{height}" fill="#f5f7fb"/>']
        self.fonts = {}

    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            filename = "arialbd.ttf" if bold else "arial.ttf"
            path = Path("C:/Windows/Fonts") / filename
            self.fonts[key] = ImageFont.truetype(str(path), size)
        return self.fonts[key]

    def text(self, x, y, value, size=22, color="#25324a", bold=False, anchor="left"):
        value = str(value)
        font = self.font(size, bold)
        tw = self.draw.textlength(value, font=font)
        dx = 0 if anchor == "left" else (-tw / 2 if anchor == "middle" else -tw)
        self.draw.text((x+dx, y), value, font=font, fill=color)
        svg_anchor = {"left": "start", "middle": "middle", "right": "end"}[anchor]
        self.svg.append(f'<text x="{x:.2f}" y="{y+size:.2f}" text-anchor="{svg_anchor}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(value)}</text>')

    def rect(self, x, y, w, h, fill, border=None):
        self.draw.rectangle((x,y,x+w,y+h), fill=fill, outline=border)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{border or "none"}"/>')

    def line(self, points, color="#25324a", width=2):
        self.draw.line(points, fill=color, width=width)
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x,y in points)
        self.svg.append(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')

    def dot(self, x, y, radius, color):
        self.draw.ellipse((x-radius,y-radius,x+radius,y+radius), fill=color)
        self.svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>')

    def save(self):
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        self.image.save(FIGURE_DIR / "模型条件演示.png")
        (FIGURE_DIR / "模型条件演示.svg").write_text("\n".join(self.svg+["</svg>"]), encoding="utf-8")


def draw_figures(results):
    c = Canvas()
    c.text(65, 32, "Two conditional models of an agent-driven information economy", 35, bold=True)
    c.text(65, 85, "Illustrative assumptions, not forecast. Independent modules; no fitted data or coupled equilibrium.", 22, color="#56637a")
    c.rect(45, 140, 780, 710, "#ffffff", "#e0e5ee")
    c.rect(850, 140, 805, 710, "#ffffff", "#e0e5ee")

    # Model 1. Crop extremely high low-N profits so the entry threshold is legible;
    # the exact N=2 and N=10 values remain in the JSON and annotation.
    c.text(75, 165, "1. Competition and the entry margin", 27, bold=True)
    c.text(75, 205, "R = 10,000; upstream share = 20%; fixed cost = 2", 20, color="#56637a")
    left, top, width, height = 125, 270, 630, 370
    xmin, xmax, ymin, ymax = 15, 100, -2, 12
    mx = lambda x: left + (x-xmin)/(xmax-xmin)*width
    my = lambda y: top + height - (y-ymin)/(ymax-ymin)*height
    entry = math.sqrt(W/FIXED)
    c.rect(mx(xmin), top, mx(entry)-mx(xmin), height, "#edf7f2")
    for y in [-2, 0, 2, 4, 6, 8, 10, 12]:
        c.line([(left,my(y)),(left+width,my(y))], "#dde3ed", 1)
        c.text(left-13, my(y)-12, y, 18, anchor="right")
    for x in [15, 30, 50, 63, 80, 100]:
        c.text(mx(x),top+height+15,x,18,anchor="middle")
    c.line([(left,top),(left,top+height),(left+width,top+height)], "#718096", 2)
    c.line([(left,my(0)),(left+width,my(0))], "#8895a9", 2)
    pts = [(mx(x),my(net_profit(x))) for x in [24+i*0.1 for i in range(761)] if ymin<=net_profit(x)<=ymax]
    c.line(pts,"#235bd7",4)
    c.line([(mx(entry),top),(mx(entry),top+height)],"#119670",2)
    c.text(mx(entry)+10, top+12, "N* = 63.25", 19, "#087b59", bold=True)
    c.dot(mx(63),my(net_profit(63)),5,"#119670")
    c.dot(mx(64),my(net_profit(64)),5,"#dc6549")
    c.text(left+width/2,top+height+55,"Independent economic decision-makers N",22,anchor="middle")
    c.text(left,top-36,"Net profit / operator / cycle",20)
    c.text(75,735,"N=63: +0.01562; N=64: -0.046875",22,bold=True)
    c.text(75,775,"Off scale: N=2 -> 1,998; N=10 -> 78.",20,color="#56637a")
    c.text(75,805,"More entry dissipates rents; this model does not imply universal loss.",18,color="#56637a")

    # Model 2.
    c.text(880,165,"2. Information quality and feedback",27,bold=True)
    c.text(880,205,"q = usable stock / (usable stock + noisy stock)",20,color="#56637a")
    left,top,width,height = 930,270,660,370
    mx = lambda t:left+t/50*width
    my = lambda q:top+height-q*height
    for q in [0,.2,.4,.6,.8,1]:
        c.line([(left,my(q)),(left+width,my(q))],"#dde3ed",1)
        c.text(left-13,my(q)-12,f"{q:.1f}",18,anchor="right")
    for t in [0,10,20,30,40,50]:
        c.text(mx(t),top+height+15,t,18,anchor="middle")
    c.line([(left,top),(left,top+height),(left+width,top+height)],"#718096",2)
    c.text(left,top-36,"Usable-information share q",20)
    c.text(left+width/2,top+height+55,"Time (abstract units)",22,anchor="middle")
    colors = ["#235bd7","#9b6ac8","#dc6549","#119670"]
    for idx,scenario in enumerate(results["information_module"]["scenarios"]):
        curve = scenario["trajectory"]
        c.line([(mx(s["t"]),my(s["q"])) for s in curve],colors[idx],4)
        row = 735+idx*27
        c.line([(880,row+13),(918,row+13)],colors[idx],4)
        a,v = scenario["activity_A"], scenario["verification_v"]
        c.text(930,row,f"A={a:g}, v={v:g}: q(50)={curve[-1]['q']:.3f}",20)
    c.text(65,885,"The noise-instability threshold depends on verification: A_c=4.5 at v=0.2; A_c=12 at v=0.6.",22)
    c.text(65,918,"Changing the assumed creation, amplification or removal mechanisms changes these outcomes.",18,color="#56637a")
    c.save()


def main():
    threshold = math.sqrt(W/FIXED)
    max_profitable = max(n for n in range(2, math.ceil(threshold)+1) if net_profit(n)>0)
    max_nonnegative = max(n for n in range(2, math.ceil(threshold)+1) if net_profit(n)>=0)
    scenarios = []
    for a,v in SCENARIOS:
        r = INFO["gamma"]*a*(1-v)-INFO["delta_B"]-INFO["eta"]*v
        source = INFO["beta"]*(1-INFO["theta"])*a
        scenarios.append({"activity_A": a, "verification_v": v,
                          "amplification_rate_r": r,
                          "critical_activity_Ac": (INFO["delta_B"]+INFO["eta"]*v)/(INFO["gamma"]*(1-v)),
                          "stable": r<0,
                          "B_equilibrium_if_stable": -source/r if r<0 else None,
                          "H_equilibrium": (INFO["S"]+INFO["alpha"]*INFO["theta"]*a)/INFO["delta_H"],
                          "trajectory": [information_state(i/2,a,v) for i in range(101)],
                          "validation": rk4_check(a,v)})
    # Explicitly test the zero-r case and a nearby case for continuity.
    exact_zero = information_state(50,4.5,.2)
    nearby = information_state(50,4.5+1e-9,.2)
    assert abs(exact_zero["B"]-(1+.12*4.5*50))<1e-10
    assert abs(nearby["B"]-exact_zero["B"])/exact_zero["B"]<1e-8
    results = {
        "scope": "Conditional mathematical demonstrations, not empirical forecasts. The two primary modules are separate and uncalibrated. The one-state feedback extension is only a counterexample; none constitute a solved full economic-information equilibrium.",
        "reproduction_command": "& 'C:\\Users\\weii2\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' 'C:\\Users\\weii2\\Documents\\Codex\\2026-10-03\\wo-x\\outputs\\模型演示.py'",
        "dependencies": "Python standard library and Pillow; no LLM or API calls.",
        "contest_module": {
            "assumptions": "Risk-neutral identical independent economic decision-makers; simultaneous effort choice; proportional contest success W*e_i/sum(e); linear effort cost; fixed active-operator cost. Multiple threads or agents controlled by the same decision-maker are not independent economic competitors and their combined effort belongs to that operator. Rival effort is positive for N>=2. Entry test evaluates the post-entry symmetric equilibrium, not transient entry dynamics.",
            "parameters": {"R": R,"tau": TAU,"F": FIXED,"c": COST,"W": W},
            "unit_convention": {"R_W_profit_F": "currency units per cycle", "N": "independent economic decision-makers, not agent threads", "effort_E": "effort units per cycle", "c": "currency per effort unit", "tau": "dimensionless upstream revenue share"},
            "formulas": {"per_agent_effort": "W*(N-1)/(c*N^2)", "total_effort": "W/c*(1-1/N)", "net_profit": "W/N^2-F", "continuous_entry_threshold": "sqrt(W/F)"},
            "continuous_entry_threshold": threshold,
            "maximum_strictly_profitable_integer": max_profitable,
            "maximum_nonnegative_integer": max_nonnegative,
            "rows": [contest_row(n) for n in [2,10,50,63,64,100]],
            "verification": "Golden-section numerical maximization of unilateral payoff against fixed equilibrium rivals; +/-10% effort deviations; accounting identity."},
        "information_module": {
            "assumptions": "A and v are exogenous constants. H and B have linear dynamics with assumed endogenous noise amplification. B denotes noisy, redundant or unreliable stock, not all AI-generated content. No monetary feedback or finite-capacity correction is imposed.",
            "parameters": INFO,
            "unit_convention": {"time": "abstract time units", "H_B": "comparable normalized information-stock units", "A": "normalized activity-pressure index held fixed", "S": "stock units per time", "alpha_beta": "stock units per time per activity index", "delta_H_delta_B_eta": "inverse time", "gamma": "inverse time per activity index", "theta_v_q": "dimensionless fractions"},
            "equations": {"H_dot": "S+alpha*theta*A-delta_H*H", "B_dot": "beta*(1-theta)*A+[gamma*A*(1-v)-delta_B-eta*v]*B", "q": "H/(H+B)", "Ac": "(delta_B+eta*v)/(gamma*(1-v)) for v<1"},
            "scenarios": scenarios,
            "zero_rate_validation": {"A":4.5,"v":.2,"t":50,"B":exact_zero["B"],"nearby_relative_difference":abs(nearby["B"]-exact_zero["B"])/exact_zero["B"]},
            "verification": "Independent RK4 integration of original ODE RHS at dt=0.01; analytic zero-rate limit and near-zero continuity."},
        "closed_feedback_counterexample": closed_feedback_counterexample(),
        "artifacts": [str(FIGURE_DIR/"模型条件演示.png"),str(FIGURE_DIR/"模型条件演示.svg")]}
    RESULT_PATH.parent.mkdir(parents=True,exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    draw_figures(results)
    print(json.dumps({"entry_threshold":threshold,"maximum_profitable_N":max_profitable,
                      "profits":{str(n):net_profit(n) for n in [2,10,50,63,64,100]},
                      "information_t50":[{"A":s["activity_A"],"v":s["verification_v"],**s["trajectory"][-1]} for s in scenarios],
                      "feedback_equilibrium":{key:results["closed_feedback_counterexample"][key] for key in ["x_equilibrium","q_equilibrium","A_equilibrium","feedback_local_derivative"]},
                      "files":[str(RESULT_PATH),*results["artifacts"]]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
