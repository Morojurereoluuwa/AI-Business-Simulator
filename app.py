from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, asdict
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
try:
    import streamlit as st
except ImportError:  # Allows the calculation tests to run before UI dependencies are installed.
    st = None

# ----------------------------- Teaching defaults -----------------------------
STATE_VERSION = 1
PATHWAY_ID = "ART-2"
COURSE = "Precalculus"
LEVELS = ("Foundation", "Applied", "Advanced")
MONEY = Decimal

DEFAULT_STATE: dict[str, Any] = {
    "team_id": "ART2-TEAM-01",
    "pathway_id": PATHWAY_ID,
    "selected_course": COURSE,
    "selected_level": "Foundation",
    "snapshot_id": "",
    "state_version": STATE_VERSION,
    "simulation_date": "2027-01-01",
    "run_id": "",
    "random_seed": 42,
    "refund_amount": "0.00",
    "loan_principal": "0.00",
    "interest": "0.00",
    "additional_capex": "0.00",
    "owner_contribution": "7000.00",
    "equipment_assets": "1800.00",
    "preopening_expense": "1200.00",
    "opening_inventory": "0.00",
    "opening_cash": "4000.00",
    "retained_earnings": "-1200.00",
    "base_price": "180.00",
    "D0": 18,
    "staff_hours": 90,
    "hours_per_unit": 6,
    "revision_hours_per_project": 0,
    "extra_revision_hourly_cost": "5.00",
    "asset_license_verified": True,
    "variable_costs": {
        "creative_labor": "25.00",
        "licensed_assets": "10.00",
        "rendering": "5.00",
        "delivery_print": "5.00",
    },
    "fixed_costs": {
        "software": "250.00",
        "workspace": "250.00",
        "admin": "150.00",
        "equipment_maintenance": "150.00",
    },
    "advertising": "0.00",
    "ledger": [],
    "approved_run_ids": [],
}

def dec(value: Any) -> Decimal:
    """Convert values to Decimal without binary-float arithmetic."""
    return value if isinstance(value, Decimal) else Decimal(str(value))

def cents(value: Any) -> Decimal:
    return dec(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def money(value: Any) -> str:
    return f"${cents(value):,.2f}"

def capacity_for(staff_hours: int, hours_per_unit: int, revision_hours_per_project: int) -> int:
    total_hours = int(hours_per_unit) + int(revision_hours_per_project)
    if total_hours <= 0:
        raise ValueError("Hours per project plus revision hours must be greater than zero.")
    if staff_hours < 0:
        raise ValueError("Staff hours cannot be negative.")
    return int(staff_hours) // total_hours

def variable_cost_per_unit(variable_costs: dict[str, Any], revision_hours: int, extra_revision_hourly_cost: Any) -> Decimal:
    base = sum((dec(v) for v in variable_costs.values()), Decimal("0"))
    return cents(base + dec(revision_hours) * dec(extra_revision_hourly_cost))

def average_cost(sold: int, fixed_expense: Any, variable_per_unit: Any) -> Decimal | None:
    if sold == 0:
        return None
    return cents(dec(fixed_expense) / Decimal(sold) + dec(variable_per_unit))

def calculate_month(
    *, requested: int, capacity: int, price: Any,
    variable_costs: dict[str, Any], fixed_costs: dict[str, Any],
    revision_hours: int = 0, extra_revision_hourly_cost: Any = "5.00",
    marketing: Any = "0.00", interest: Any = "0.00",
    depreciation: Any = "0.00", opening_cash: Any = "4000.00",
    principal: Any = "0.00", capex: Any = "0.00",
    new_owner_cash: Any = "0.00", loan_cash: Any = "0.00",
) -> dict[str, Any]:
    """One-period model. All money calculations use Decimal."""
    sold = min(max(0, int(requested)), max(0, int(capacity)))
    unit_variable = variable_cost_per_unit(variable_costs, revision_hours, extra_revision_hourly_cost)
    revenue = cents(dec(price) * sold)
    variable_expense = cents(unit_variable * sold)
    fixed_expense = cents(sum((dec(v) for v in fixed_costs.values()), Decimal("0")))
    marketing = cents(marketing)
    interest = cents(interest)
    depreciation = cents(depreciation)
    operating_profit = cents(revenue - variable_expense - fixed_expense - marketing)
    net_profit = cents(operating_profit - interest - depreciation)
    total_expense = cents(variable_expense + fixed_expense + marketing + interest + depreciation)
    ending_cash = cents(
        dec(opening_cash) + revenue - variable_expense - fixed_expense - marketing
        - interest - dec(principal) - dec(capex) + dec(new_owner_cash) + dec(loan_cash)
    )
    return {
        "sold": sold, "revenue": revenue, "unit_variable": unit_variable,
        "variable_expense": variable_expense, "fixed_expense": fixed_expense,
        "marketing": marketing, "operating_profit": operating_profit,
        "net_profit": net_profit, "total_expense": total_expense,
        "ending_cash": ending_cash,
    }

def seasonal_demand(month_index: int, D0: Any) -> float:
    """Assumed seasonal pattern: period 12, peak at month 4, low at month 10."""
    return float(D0) * (1 + 0.20 * math.sin(2 * math.pi * (int(month_index) - 1) / 12))

def demand_at_price(price: float, p0: float, D0: float) -> float:
    if p0 <= 0:
        raise ValueError("Base price must be positive.")
    return max(0.0, 2 * D0 - (D0 / p0) * price)

def build_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(state, default=str))
    payload["state_version"] = STATE_VERSION
    payload["snapshot_id"] = payload.get("snapshot_id") or str(uuid.uuid4())
    return payload

def validate_snapshot(snapshot: dict[str, Any], expected_team_id: str) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise ValueError("Snapshot must be a JSON object.")
    if snapshot.get("state_version") != STATE_VERSION:
        raise ValueError(f"Stale or unsupported state_version. Expected {STATE_VERSION}.")
    if snapshot.get("team_id") != expected_team_id:
        raise ValueError("Snapshot team_id does not match the active team.")
    required = {"opening_cash", "owner_contribution", "equipment_assets", "preopening_expense",
                "opening_inventory", "retained_earnings", "ledger"}
    if not required.issubset(snapshot):
        raise ValueError("Snapshot is missing required balance or ledger fields.")
    return snapshot

def run_fixture(fixture: pd.DataFrame, state: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply fixture rows sequentially, using capacity_override only in fixture mode."""
    cash = dec(state["opening_cash"])
    results, transactions = [], []
    for row in fixture.to_dict("records"):
        result = calculate_month(
            requested=int(row["requested"]),
            capacity=int(row["capacity_override"]),
            price=state["base_price"],
            variable_costs=state["variable_costs"],
            fixed_costs=state["fixed_costs"],
            revision_hours=int(state["revision_hours_per_project"]),
            extra_revision_hourly_cost=state["extra_revision_hourly_cost"],
            marketing=row["ad_cost"],
            opening_cash=cash,
        )
        cash = result["ending_cash"]
        results.append({
            "month": row["month"], "requested": int(row["requested"]),
            "capacity": int(row["capacity_override"]), "sold": result["sold"],
            "revenue": result["revenue"], "total_expense": result["total_expense"],
            "profit": result["net_profit"], "closing_cash": result["ending_cash"],
        })
        transactions.append({
            "date": f'{row["month"]}-28', "type": "monthly_simulation",
            "run_id": f'fixture-{row["month"]}', "revenue": str(result["revenue"]),
            "variable_expense": str(result["variable_expense"]),
            "fixed_expense": str(result["fixed_expense"]),
            "marketing": str(result["marketing"]), "interest": "0.00",
            "principal": "0.00", "capex": "0.00", "sold": result["sold"],
        })
    return pd.DataFrame(results), transactions

def ledger_totals(ledger: list[dict[str, Any]]) -> dict[str, Decimal]:
    keys = ("revenue", "variable_expense", "fixed_expense", "marketing", "interest", "principal", "capex")
    return {key: cents(sum((dec(tx.get(key, "0")) for tx in ledger), Decimal("0"))) for key in keys}

def add_ledger_transaction(state: dict[str, Any], transaction: dict[str, Any]) -> bool:
    """Append once per run_id. Returns False for duplicate run IDs."""
    run_id = transaction.get("run_id")
    if not run_id:
        raise ValueError("Every ledger transaction needs a run_id.")
    existing = {tx.get("run_id") for tx in state["ledger"]}
    if run_id in existing or run_id in state["approved_run_ids"]:
        return False
    state["ledger"].append(transaction)
    state["approved_run_ids"].append(run_id)
    return True


def ledger_period_summary(ledger: list[dict[str, Any]], period: str) -> pd.DataFrame:
    """Aggregate the same dated ledger into daily, monthly, or yearly views."""
    if not ledger:
        return pd.DataFrame(columns=["period", "transactions", "revenue", "expenses", "net"])
    frame = pd.DataFrame(ledger).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    if period == "Daily":
        frame["period"] = frame["date"].dt.strftime("%Y-%m-%d")
    elif period == "Monthly":
        frame["period"] = frame["date"].dt.strftime("%Y-%m")
    else:
        frame["period"] = frame["date"].dt.strftime("%Y")
    for col in ["revenue", "variable_expense", "fixed_expense", "marketing", "interest"]:
        if col not in frame:
            frame[col] = "0"
        frame[col] = frame[col].map(lambda x: float(dec(x)))
    frame["expenses"] = frame[["variable_expense", "fixed_expense", "marketing", "interest"]].sum(axis=1)
    frame["net"] = frame["revenue"] - frame["expenses"]
    return (frame.groupby("period", as_index=False)
            .agg(transactions=("run_id", "count"), revenue=("revenue", "sum"),
                 expenses=("expenses", "sum"), net=("net", "sum"))
            .sort_values("period"))

# -------------------------------- Streamlit UI --------------------------------
def init_state() -> None:
    if "business_state" not in st.session_state:
        st.session_state.business_state = json.loads(json.dumps(DEFAULT_STATE))
    if "forecast" not in st.session_state:
        st.session_state.forecast = None

def fmt_table_money(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out:
            out[col] = out[col].map(money)
    return out

def render_math_panel(state: dict[str, Any]) -> None:
    st.subheader("Precalculus Math Skills")
    st.caption("The selected level controls which mathematical model is displayed. All values are teaching assumptions, not market forecasts.")
    level = state["selected_level"]
    p0, D0 = float(dec(state["base_price"])), float(state["D0"])
    cap = capacity_for(state["staff_hours"], state["hours_per_unit"], state["revision_hours_per_project"])
    fixed = sum((dec(x) for x in state["fixed_costs"].values()), Decimal("0")) + dec(state["advertising"])
    v = variable_cost_per_unit(state["variable_costs"], state["revision_hours_per_project"], state["extra_revision_hourly_cost"])
    st.markdown(f"**Course:** {state['selected_course']}  |  **Teacher-approved level:** {level}")
    if level == "Foundation":
        st.markdown("**Skill:** Rational functions and asymptotes")
        st.markdown("**Business question:** How does average cost per project change as monthly output increases?")
        st.latex(r"A(s)=\frac{F}{s}+v,\quad 0<s\\leq\\text{capacity}")
        st.write(f"Variables: s = projects/month; F = fixed monthly expense ({money(fixed)}); v = variable cost/project ({money(v)}).")
        st.write(f"Domain: 0 < s ≤ {cap} projects. Assumptions: fixed cost and unit variable cost stay constant; only feasible monthly output is shown.")
        st.info("At s = 0, A(0) is undefined (vertical asymptote s = 0). The horizontal asymptote is y = v = " + money(v) + ". This is a mathematical limit; the practical domain ends at monthly capacity.")
        xs = list(range(1, max(cap, 1) + 1))
        ys = [float(fixed / Decimal(x) + v) for x in xs]
        fig, ax = plt.subplots()
        ax.plot(xs, ys, marker="o", label="Average cost")
        ax.axhline(float(v), linestyle="--", label=f"Horizontal asymptote y={float(v):.2f}")
        ax.set(xlabel="Projects per month (s)", ylabel="Average cost ($/project)", title="Foundation: Average Cost")
        ax.legend()
        st.pyplot(fig)
        table = pd.DataFrame({"s (projects)": xs, "A(s) ($/project)": [money(y) for y in ys]})
        st.dataframe(table, use_container_width=True)
        st.write(f"At s = 0: **undefined**. At s = 15: **{money(average_cost(15, fixed, v))}** (if 15 is within capacity).")
    elif level == "Applied":
        st.markdown("**Skill:** Function composition, piecewise functions, and optimization")
        st.markdown("**Business question:** How does changing the project price affect demand, completed sales, revenue, cost, and profit?")
        st.latex(r"q(p)=\max\left(0,2D_0-\frac{D_0}{p_0}p\right)")
        st.latex(r"s(p)=\\min(\\text{capacity},q(p)),\quad R(p)=p\\,s(p)")
        st.latex(r"C(s(p))=F+v\\,s(p),\quad P(p)=R(p)-C(s(p))")
        st.write(f"Input-output mapping: price p ($/project) → requested demand q(p) (projects) → completed sales s(p) (projects) → revenue R(p), cost C, and profit P ($).")
        st.write(f"Variables: p₀ = ${p0:.2f}; D₀ = {D0:g} projects/month; capacity = {cap}; F = {money(fixed)}; v = {money(v)}.")
        st.write(f"Domain: 0 ≤ p ≤ {money(2*p0)}. Assumptions: linear demand is an illustrative model; completed sales cannot exceed capacity; integer outcomes are rounded down for operational planning.")
        breakpoint = p0 * (2 - cap / D0) if D0 > 0 else float("nan")
        if D0 > 0 and 0 <= breakpoint <= 2*p0:
            st.info(f"Capacity breakpoint: p = p₀(2 − capacity/D₀) = ${breakpoint:.2f}. Below this price, modeled demand is at/above capacity.")
        prices = [2*p0*i/100 for i in range(101)]
        demand = [demand_at_price(p, p0, D0) for p in prices]
        sales_cont = [min(cap, q) for q in demand]
        revenue = [p*s for p, s in zip(prices, sales_cont)]
        cost = [float(fixed) + float(v)*s for s in sales_cont]
        profit = [r-c for r,c in zip(revenue,cost)]
        fig, ax = plt.subplots()
        ax.plot(prices, revenue, label="Revenue R(p)")
        ax.plot(prices, cost, label="Cost C(s(p))")
        ax.plot(prices, profit, label="Profit P(p)")
        ax.set(xlabel="Price ($/project)", ylabel="Dollars per month", title="Applied: Continuous price model")
        ax.legend()
        st.pyplot(fig)
        sample_prices = sorted(set([0, round(breakpoint, 2) if math.isfinite(breakpoint) else p0, p0, 1.5*p0, 2*p0]))
        rows = []
        for p in sample_prices:
            q = demand_at_price(p, p0, D0)
            sold_int = min(cap, max(0, math.floor(q + 1e-9)))
            rev = cents(Decimal(str(p))*sold_int)
            cost_int = cents(fixed + v*sold_int)
            rows.append({"Price ($)": money(p), "Demand q(p) continuous": round(q, 2),
                         "Completed sales (integer)": sold_int, "Revenue ($)": money(rev),
                         "Cost ($)": money(cost_int), "Profit ($)": money(rev-cost_int)})
        st.caption("Continuous curves use fractional modeled demand; the table shows whole-project operating outcomes.")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.markdown("**Skill:** Periodic functions (seasonal demand) — assumed seasonality")
        st.markdown("**Business question:** How might an assumed seasonal pattern change monthly demand before capacity limits?")
        st.latex(r"D(t)=D_0\left(1+0.20\sin\left(\frac{2\pi(t-1)}{12}\right)\right)")
        st.write(f"Variables: t = month index (1–12); D₀ = baseline demand ({D0:g} projects/month). Period = 12 months; peak at t = 4; minimum at t = 10.")
        st.write(f"Domain: integer month index t = 1,…,12 (the sine curve can be extended periodically). Capacity = {cap} projects/month. Assumption: ±20% seasonality is invented for classroom modeling, not observed market data.")
        months = list(range(1,13))
        raw = [seasonal_demand(t,D0) for t in months]
        clipped = [min(cap,x) for x in raw]
        fig, ax = plt.subplots()
        ax.plot(months, raw, marker="o", label="Demand before capacity")
        ax.plot(months, clipped, marker="s", label="Demand clipped at capacity")
        ax.axhline(cap, linestyle="--", label="Capacity")
        ax.set(xlabel="Month index t", ylabel="Projects", title="Advanced: Assumed seasonal demand")
        ax.set_xticks(months)
        ax.legend()
        st.pyplot(fig)
        st.dataframe(pd.DataFrame({"Month t": months, "Demand before capacity": [round(x,2) for x in raw],
                                   "Completed sales after capacity": [min(cap, math.floor(x)) for x in raw]}), use_container_width=True)
        st.write(f"Check: month 4 = {seasonal_demand(4,D0):.2f} ({1.2*D0:.2f} expected); month 10 = {seasonal_demand(10,D0):.2f} ({0.8*D0:.2f} expected).")

def main() -> None:
    if st is None:
        raise RuntimeError('Streamlit is required to launch the app. Run: pip install -r requirements.txt')
    st.set_page_config(page_title="Career Venture Lab | ART-2", page_icon="🎨", layout="wide")
    init_state()
    state = st.session_state.business_state
    st.title("Career Venture Lab")
    st.caption("Pathway ART-2 · Precalculus · Design and Animation Studio · Stage 1")
    page = st.sidebar.radio("Stage 1 modules", ["Overview", "Inputs & Funding", "Math Skills & Levels", "Save / Load JSON"])
    if page == "Overview":
        st.header("Studio overview")
        st.write("A fictional design-and-animation studio used to explore capacity, pricing, costs, profit, and cash flow.")
        c1,c2,c3,c4 = st.columns(4)
        cap = capacity_for(state["staff_hours"], state["hours_per_unit"], state["revision_hours_per_project"])
        c1.metric("Opening spendable cash", money(state["opening_cash"]))
        c2.metric("Monthly capacity", f"{cap} projects")
        c3.metric("Base price", money(state["base_price"]))
        c4.metric("Baseline demand (D₀)", f'{state["D0"]} projects')
        st.subheader("Opening balance sheet snapshot")
        opening = pd.DataFrame([
            ("Owner contribution", money(state["owner_contribution"])),
            ("Equipment assets (asset, not expense)", money(state["equipment_assets"])),
            ("Pre-opening expense", money(state["preopening_expense"])),
            ("Opening inventory", money(state["opening_inventory"])),
            ("Post-startup spendable cash", money(state["opening_cash"])),
            ("Retained earnings", money(state["retained_earnings"])),
        ], columns=["Item","Amount"])
        st.dataframe(opening, hide_index=True, use_container_width=True)
        st.caption("Equipment and inventory are not expensed again. Funding and equipment purchases are not sales or operating expenses.")
        st.subheader("Baseline monthly operating structure")
        col1,col2 = st.columns(2)
        with col1:
            st.write("**Variable cost per project**")
            st.dataframe(pd.DataFrame([{"Cost line":k.replace("_"," ").title(),"Amount":money(v)} for k,v in state["variable_costs"].items()]), hide_index=True)
            st.metric("Total variable cost / project", money(variable_cost_per_unit(state["variable_costs"], state["revision_hours_per_project"], state["extra_revision_hourly_cost"])))
        with col2:
            st.write("**Monthly fixed costs**")
            st.dataframe(pd.DataFrame([{"Cost line":k.replace("_"," ").title(),"Amount":money(v)} for k,v in state["fixed_costs"].items()]), hide_index=True)
            st.metric("Total fixed cost / month", money(sum((dec(x) for x in state["fixed_costs"].values()), Decimal("0"))))
        st.write(f"Advertising campaign line: **{money(state['advertising'])}** (separate from fixed costs).")
        if state["ledger"]:
            st.subheader("Recorded ledger activity")
            totals = ledger_totals(state["ledger"])
            st.write("Transactions are dated and shared by all reporting periods. Forecasts remain separate until approved.")
            st.dataframe(pd.DataFrame(state["ledger"]), use_container_width=True)
            st.write("Ledger totals:", {k: money(v) for k,v in totals.items()})
            period = st.selectbox("Ledger reporting view", ["Daily", "Monthly", "Yearly"])
            report = ledger_period_summary(state["ledger"], period)
            st.dataframe(fmt_table_money(report, ["revenue", "expenses", "net"]), hide_index=True, use_container_width=True)
        else:
            st.info("No actual transactions have been approved yet. Forecasts do not change the ledger.")
    elif page == "Inputs & Funding":
        st.header("Inputs & Funding")
        st.caption("Edit the fictional teaching defaults. Normal-mode capacity is derived from available staff hours.")
        with st.form("inputs_form"):
            left,right = st.columns(2)
            with left:
                state["team_id"] = st.text_input("Team ID", state["team_id"])
                state["selected_level"] = st.selectbox("Selected level", LEVELS, index=LEVELS.index(state["selected_level"]))
                state["simulation_date"] = st.date_input("Simulation date", date.fromisoformat(state["simulation_date"])).isoformat()
                state["base_price"] = f"{st.number_input('Base price ($ / project)', min_value=0.01, value=float(dec(state['base_price'])), step=5.0):.2f}"
                state["D0"] = st.number_input("Baseline monthly demand D₀ (projects)", min_value=0, value=int(state["D0"]), step=1)
                state["staff_hours"] = st.number_input("Staff hours / month", min_value=0, value=int(state["staff_hours"]), step=5)
                state["hours_per_unit"] = st.number_input("Hours per project", min_value=1, value=int(state["hours_per_unit"]), step=1)
                state["revision_hours_per_project"] = st.number_input("Extra revision hours / project", min_value=0, value=int(state["revision_hours_per_project"]), step=1)
                state["extra_revision_hourly_cost"] = f"{st.number_input('Extra revision hourly cost ($)', min_value=0.0, value=float(dec(state['extra_revision_hourly_cost'])), step=1.0):.2f}"
                state["asset_license_verified"] = st.checkbox("Licensed assets verified", value=bool(state["asset_license_verified"]))
            with right:
                st.write("**Funding and opening balances**")
                for key,label in [("owner_contribution","Owner contribution"),("equipment_assets","Equipment assets"),("preopening_expense","Pre-opening expense"),("opening_inventory","Opening inventory"),("opening_cash","Spendable opening cash"),("retained_earnings","Retained earnings"),("loan_principal","Loan principal"),("interest","Interest"),("refund_amount","Refund amount"),("additional_capex","Additional capital expenditure")]:
                    state[key] = f"{st.number_input(label+' ($)', value=float(dec(state[key])), step=100.0):.2f}"
                st.write("**Variable cost lines ($ / project)**")
                for key in state["variable_costs"]:
                    state["variable_costs"][key] = f"{st.number_input(key.replace('_',' ').title(), min_value=0.0, value=float(dec(state['variable_costs'][key])), step=1.0, key='v_'+key):.2f}"
                st.write("**Fixed monthly cost lines ($)**")
                for key in state["fixed_costs"]:
                    state["fixed_costs"][key] = f"{st.number_input(key.replace('_',' ').title(), min_value=0.0, value=float(dec(state['fixed_costs'][key])), step=25.0, key='f_'+key):.2f}"
                state["advertising"] = f"{st.number_input('Advertising campaign ($ / month)', min_value=0.0, value=float(dec(state['advertising'])), step=25.0):.2f}"
            submitted = st.form_submit_button("Save input changes")
        if submitted:
            st.session_state.business_state = state
            st.success("Inputs saved for this session.")
        cap = capacity_for(state["staff_hours"], state["hours_per_unit"], state["revision_hours_per_project"])
        st.metric("Derived monthly capacity", f"{cap} projects")
        st.subheader("Forecast / what-if (not recorded)")
        st.write("Adjust a scenario. It will not affect actual ledger balances unless you approve the dated transaction.")
        with st.form("forecast_form"):
            requested = st.slider("Requested projects", 0, max(30, int(state["D0"])*3), min(int(state["D0"]), max(30, int(state["D0"])*3)))
            marketing = st.number_input("Scenario advertising ($)", min_value=0.0, value=float(dec(state["advertising"])), step=25.0)
            run_id = st.text_input("Run ID (unique for approval)", value=f"run-{state['simulation_date']}-{requested}-{int(marketing)}")
            forecast_button = st.form_submit_button("Calculate forecast")
        if forecast_button:
            st.session_state.forecast = {
                "requested": requested, "marketing": f"{marketing:.2f}", "run_id": run_id,
                "date": state["simulation_date"],
            }
        if st.session_state.forecast:
            f = st.session_state.forecast
            result = calculate_month(requested=f["requested"], capacity=cap, price=state["base_price"],
                variable_costs=state["variable_costs"], fixed_costs=state["fixed_costs"],
                revision_hours=state["revision_hours_per_project"],
                extra_revision_hourly_cost=state["extra_revision_hourly_cost"],
                marketing=f["marketing"], opening_cash=state["opening_cash"])
            st.write({k: money(v) if isinstance(v, Decimal) else v for k,v in result.items()})
            if not state["asset_license_verified"]:
                st.warning("Publishing is blocked: verify licensed assets before publishing a draft.")
            if st.button("Approve and record this forecast", disabled=not state["asset_license_verified"]):
                tx = {"date": f["date"], "type":"approved_monthly_run", "run_id":f["run_id"],
                      "requested":f["requested"], "sold":result["sold"], "revenue":str(result["revenue"]),
                      "variable_expense":str(result["variable_expense"]), "fixed_expense":str(result["fixed_expense"]),
                      "marketing":str(result["marketing"]), "interest":"0.00", "principal":"0.00", "capex":"0.00"}
                if add_ledger_transaction(state, tx):
                    state["opening_cash"] = str(result["ending_cash"])
                    st.session_state.business_state = state
                    st.success("Approved once and added to the dated ledger.")
                else:
                    st.warning("Duplicate run_id blocked; no second transaction was recorded.")
        fixture_mode = st.checkbox("Enable Fixture-test mode (sample data only)", value=False)
        if fixture_mode:
            st.warning("FIXTURE-TEST MODE: capacity_override from sample_data/monthly_fixture.csv is being used for these teaching checks. These rows are not actual transactions and will not change cash or the ledger. Normal mode derives capacity from staff hours.")
            fixture_path = Path(__file__).parent / "sample_data" / "monthly_fixture.csv"
            fixture_df = pd.read_csv(fixture_path)
            fixture_results, _ = run_fixture(fixture_df, state)
            st.dataframe(fmt_table_money(fixture_results, ["revenue", "total_expense", "profit", "closing_cash"]), hide_index=True, use_container_width=True)
        else:
            st.caption("Normal mode: capacity is derived from staff_hours ÷ (hours_per_unit + revision_hours_per_project). Fixture capacity overrides are not used.")
    elif page == "Math Skills & Levels":
        st.header("Math Skills & Levels")
        st.selectbox("Teacher-approved level", LEVELS, key="math_level", index=LEVELS.index(state["selected_level"]))
        state["selected_level"] = st.session_state.math_level
        render_math_panel(state)
    else:
        st.header("Save / Load JSON snapshot")
        st.write("Export preserves balances, controls, and the dated ledger. Imports must match this team and the current state version.")
        state["snapshot_id"] = state.get("snapshot_id") or str(uuid.uuid4())
        payload = build_snapshot(state)
        st.download_button("Download JSON snapshot", json.dumps(payload, indent=2), file_name="career_venture_lab_snapshot.json", mime="application/json")
        st.caption(f"Current snapshot ID: {payload['snapshot_id']} · state_version: {STATE_VERSION} · team: {state['team_id']}")
        uploaded = st.file_uploader("Import a JSON snapshot", type=["json"])
        if uploaded is not None:
            try:
                incoming = json.loads(uploaded.getvalue().decode("utf-8"))
                validated = validate_snapshot(incoming, state["team_id"])
                st.session_state.business_state = validated
                st.session_state.forecast = None
                st.success("Snapshot loaded. Balances and ledger restored.")
                st.rerun()
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                st.error(f"Import rejected: {exc}")

if __name__ == "__main__":
    main()
