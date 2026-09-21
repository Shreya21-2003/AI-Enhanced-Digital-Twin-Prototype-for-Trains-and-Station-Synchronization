import pulp

def solve_platform_rerouting(train_id: str, conflicting_platform: str, available_platforms: list, arrival_delays: dict):
    """
    Mixed-Integer Linear Programming (MILP) solver for conflict-free platform allocation.
    Minimizes total passenger dwell delay while enforcing interlocking exclusivity.
    """
    prob = pulp.LpProblem("Platform_Dispatch_Optimization", pulp.LpMinimize)

    # Decision variables: x[p] = 1 if train is routed to platform p, 0 otherwise
    x = {p: pulp.LpVariable(f"route_{p}", cat='Binary') for p in available_platforms}

    # Objective: Minimize arrival and secondary walking delay
    prob += pulp.lpSum([arrival_delays[p] * x[p] for p in available_platforms])

    # Constraint 1: Must assign exactly one platform
    prob += pulp.lpSum([x[p] for p in available_platforms]) == 1

    # Constraint 2: Conflicted/blocked platform is strictly forbidden
    if conflicting_platform in x:
        prob += x[conflicting_platform] == 0

    # Solve via Branch-and-Bound
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    for p in available_platforms:
        if pulp.value(x[p]) == 1.0:
            return {
                "assigned_platform": p,
                "status": "OPTIMAL",
                "delay_penalty_sec": arrival_delays[p] * 60
            }
    return {"assigned_platform": None, "status": "INFEASIBLE"}