# TheMergeConflict
## Files Overview

### `practice.jl`
- **Purpose**: Solves SWE as a DAE using **implicit time integration**.
- **Boundary Conditions**: Includes support for **Dirichlet** and **Neumann** conditions.
- **Features**:
  - Topography and bed friction
  - Explicit handling of physical boundaries
  - Suitable for realistic inflow/outflow or closed domain scenarios

### `solution.jl`
- **Purpose**: Simpler version for periodic domains.
- **Boundary Conditions**: Uses **periodic** boundary conditions via `circshift`.
- **Features**:
  - Clean baseline for symmetric wave propagation
  - Great for testing convergence, conservation, and oscillatory behavior
