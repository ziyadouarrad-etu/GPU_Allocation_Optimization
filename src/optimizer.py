from pulp import *

def solve_gpu_allocation(costs, time_matrix, availability, demands):
    """
    Solves the GPU allocation problem using Linear Programming.
    """
    # Initialize the minimization problem
    problem = LpProblem("Minimize_GPU_Operational_Cost", LpMinimize)
    
    # Decision Variables: x_i_j (Integer units of model i on GPU j)
    num_models = len(demands)
    num_gpus = len(availability)
    x = [[LpVariable(f"x_{i}_{j}", lowBound=0, cat='Integer') 
          for j in range(num_gpus)] for i in range(num_models)]

    # Objective Function: Minimize total cost
    problem += lpSum(costs[i][j] * x[i][j] 
                     for i in range(num_models) for j in range(num_gpus))

    # Constraints: Availability (GPU Time)
    for j in range(num_gpus):
        problem += lpSum(x[i][j] * time_matrix[i][j] 
                         for i in range(num_models)) <= availability[j], f"GPU_{j}_Limit"

    # Constraints: Demand (Model units)
    for i in range(num_models):
        problem += lpSum(x[i][j] 
                         for j in range(num_gpus)) == demands[i], f"Model_{i}_Demand"

    # Solve
    problem.solve(PULP_CBC_CMD(msg=0))
    
    # Extract results
    allocation = [[x[i][j].varValue for j in range(num_gpus)] for i in range(num_models)]
    return LpStatus[problem.status], allocation, value(problem.objective)