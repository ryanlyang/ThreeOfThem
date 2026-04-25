import torch

G = 9.8

class PhysicsEngine:
    def __init__(self, target_orbit, m, 
        dT=1.0, dt = .01, 
        initial_x = None, initial_v= None,
        max_fuel=10, radii=torch.tensor([1,1,1])):
        """
        Initializes the simulator.

        target_orbit: (X, V) where N is the number of samples from the target orbit
            -> X: [N, 3, 3]     - Points in the target orbit
            -> V: [N, 3, 3]     - Velocities at every sample point of the target orbit (Ensures)
        m: [3]  - Masses of the three bodies
        dT: Agent action step size - time agent thrust will be applied for
        dt: Simulator integrator step-size - smaller step-size increases simulation accuracy
        intial_x: [3, 3]  - Initial positions for the bodies
        intial_v: [3,3]  - Initial velocities for the bodies
        max_fuel: maximum amount of fuel desired per action
        radii: Radii of the thee bodies
        """
        self.dT = dT
        self.dt = dt
        self.target_orbit = target_orbit
        self.max_fuel = max_fuel
        # Weights for each of the components in the reward function
        self.w1_x, self.w1_v, self.w2, self.w3 = .4, .1, .3, .2
        
        self.x = (initial_x if initial_x is not None 
            else torch.rand((3,3), dtype=torch.float64))
        self.v = (initial_v if initial_v is not None
            else torch.rand((3,3), dtype=torch.float64))
        
        self.m = m
        # Order permuations of the mass array for easy access later
        self.m_10 = m[1,2,0]
        self.m_20 = m[2,0,1]

        self.radii = radii

    def reset(self):
        """
        Resets the universe to a random, slightly unstable initial configuration.
        Returns the initial graph state (nodes and edges).
        """
        # TODO: Initialize self.positions, self.velocities, self.masses
        
        return self._get_graph_state()

    def _get_graph_state(self):
        """
        Converts raw physics arrays into PyTorch tensors for the GNN.
        Returns node_features (pos, vel, mass) and edge_features (relative distances).
        """
        # TODO: Construct and return the graph dictionary or PyTorch Geometric Data object
        pass

    def _compute_derivs(self, x, v):
        x_10, x_20 = (x - x[:,[1,2,0]]), (x - x[:,[2,0,1]])
        v_10, v_20 = (v - v[:,[1,2,0]]), (v - v[:,[2,0,1]])

        x_10_norm, x_20_norm = (torch.sqrt(torch.sum(x_10 * x_10, dim=0)),
            torch.sqrt(torch.sum(x_20 * x_20, dim=0)))
        
        a_g = G * (((1 / torch.pow(x_10_norm.view(-1, 1), 3)) * (self.m_10.view(-1,1) * x_10))
                + ((1 / torch.pow(x_20_norm.view(-1,1), 3)) * (self.m_20.view(-1,1) * x_20)))
        j = G * ((((1 / torch.pow(x_10_norm.view(-1,1), 3)) * v_10) - 
                    (1 / torch.pow(x_10_norm.view(-1,1), 5))) * (self.m_10.view(-1,1)
                    * ((torch.sum(v_10 * x_10, dim=0)) * x_10)) +
                (((1 / torch.pow(x_10_norm.view(-1,1), 3)) * v_10) - 
                    (1 / torch.pow(x_10_norm.view(-1,1), 5))) * (self.m_10.view(-1,1)
                    * ((torch.sum(v_10 * x_10, dim=0) * x_10))))
        
        return a_g, j
    def _sim_step(self, a_t):
        """
        Computes next state of the system after agent actions using the 4th-order Hermite scheme
        The simulation is stepped up to dT time forward in dt step-size time-steps
        
        a_t: [3,3]  - Accelerations applied by the agents to each body respectively
        """
        
        K = self.dT // self.dt

        for _ in range(K):
            a_g_0, j_0 = self._compute_derivs(self, self.x, self.v)
            a_0 = a_g_0 + a_t

            x_p = self.x + self.v * self.dt + (1 / 2) * a_0 * (self.dt**2) + (1 / 6) * j_0 * (self.dt**3)
            v_p = self.v + a_0 * self.dt + (1 / 2) * j_0 * (self.dt**2)

            a_g_p, j_p = self._compute_derivs(self, x_p, v_p)
            a_p = a_g_p + a_t

            v_t = (1 / 2) * (a_0 + a_p) * self.dt + (1 / 12) * (j_0 - j_p) * (self.dt**2)
            self.x += (self.v + (1 / 2) * v_t) * self.dt + (1 / 12) * (a_0 - a_p) * (self.dt**2)
            self.v += v_t

    def _check_collision(self):
        """
        Checks whether a collition between any two of the bodies has occured
        """

        x_10 , x_20 = (self.x - self.x[:,[1,2,0]]), (self.x - self.x[:,[2,0,1]])
        x_10_norm, x_20_norm = (torch.sqrt(torch.sum(x_10 * x_10, dim=0)),
            torch.sqrt(torch.sum(x_20 * x_20, dim=0)))
        
        r_10, r_20 = ((self.radii + self.radii[[1,2,0]]),
            (self.radii + self.radii[[2,0,1]]))
        
        return torch.any(x_10_norm < r_10) or torch.any(x_20_norm < r_20)

    def step(self, a_t):
        """
        Executes one MARL Macro-step.

        a_t: [3, 3]  - Acceleration vectors to be applied to each of the bodies
        """
        self._sim_step(a_t)
        
        X_o, V_o = self.target_orbit
        X_o, V_o = torch.flatten(X_o, -2), torch.flatten(V_o, -2)
        x_flat, v_flat = torch.flatten(self.x), torch.flatten(self.v)
        x_dist = torch.cdist(x_flat, X_o)
        k = torch.argmin(x_dist)
        
        R_orbit = -(self.w1_x * x_dist[k]**2 + self.w1_v * (v_flat @ V_o[k]))
        R_fuel = torch.log(torch.sum(a_t * a_t) / self.max_fuel)
        R_survive = -100 if self._check_collision else .1
        
        reward = R_orbit + self.w2 * R_fuel + self.w3 * R_survive
        done = R_survive < 0
        
        return self._get_graph_state(), reward, done, {}