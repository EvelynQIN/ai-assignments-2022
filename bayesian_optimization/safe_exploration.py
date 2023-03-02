import scipy.ndimage
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

domain = np.array([[0, 5]])

""" Solution """


class BO_algo():

    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        # pass
        self.grid = 5 * 512  # 5*256 5*128
        self.x_grid, self.step = np.linspace(start=domain[0, 0], stop=domain[0, 1], num=self.grid, retstep=True, endpoint=True)

        # safe region mask, maximization mask, exploration mask
        self.S, self.M, self.G = [], [], []

        # lower and upper bound of f and v
        self.lf, self.uf, self.lv, self.uv = [], [], [], []

        # record observed value
        self.x_ob, self.f_ob, self.v_ob = [], [], []

        # hyperparameter
        self.L = 5.0
        self.p = 1.0  # p of p-norm that measure the distance of d(x,y) = |x-y|^p
        self.beta = 2.0

        self.threshold = 1.201

        # kernel and others
        self.f_mean = 0
        self.f_kernel = 0.5 * Matern(length_scale=0.5, nu=2.5)
        self.f_noise = WhiteKernel(noise_level=0.15)

        self.kf_xx = self.f_kernel(self.x_grid.reshape(-1, 1))  # this is reused many times

        self.v_mean = 1.5
        self.v_kernel = np.sqrt(2.0) * Matern(length_scale=0.5, nu=2.5)
        self.v_noise = WhiteKernel(noise_level=0.0001)

        self.kv_xx = self.v_kernel(self.x_grid.reshape(-1, 1)) # this is reused many times

        # store for lipstriz distance L*d(x,x') for future use
        diff = np.abs(self.x_grid.reshape(-1, 1) - self.x_grid.reshape(1, -1))
        self.lp_dist = self.L * np.power(diff, self.p)

        # self.n_violation = 0
        # self.change_time = 0

    def confidence_on_grid(self):
        """
        Return the lower bound and upper bound of each grid x
        and also update the kernel cache to newest
        """
        X = np.array(self.x_ob).reshape(-1, 1) # column vector
        F = np.array(self.f_ob).reshape(-1, 1) # column vector
        V = np.array(self.v_ob).reshape(-1, 1) # column vector

        x = self.x_grid.reshape(-1, 1)

        # update the posterior
        self.kf_XX = self.f_kernel(X) + self.f_noise(X)  # NxN
        self.k_inv_f = np.linalg.solve(self.kf_XX, F - self.f_mean)  # Nx1
        kf_xX = self.f_kernel(x, X)  # n_G x N

        mean_f = self.f_mean + (kf_xX @ self.k_inv_f).reshape(-1)
        cov_f = self.kf_xx - kf_xX @ np.linalg.solve(self.kf_XX, kf_xX.T)
        sigma_f = np.sqrt(np.diagonal(cov_f))

        l_f = mean_f - np.sqrt(self.beta) * sigma_f
        u_f = mean_f + np.sqrt(self.beta) * sigma_f
        w_f = 2 * np.sqrt(self.beta) * sigma_f

        self.kv_XX = self.v_kernel(X) + self.v_noise(X)  # NxN
        self.k_inv_v = np.linalg.solve(self.kv_XX, V - self.v_mean)  # Nx1
        kv_xX = self.v_kernel(x, X)  # n_G x N

        mean_v = self.v_mean + (kv_xX @ self.k_inv_v).reshape(-1)
        cov_v = self.kv_xx - kv_xX @ np.linalg.solve(self.kv_XX, kv_xX.T)
        sigma_v = np.sqrt(np.diagonal(cov_v))

        l_v = mean_v - np.sqrt(self.beta) * sigma_v
        u_v = mean_v + np.sqrt(self.beta) * sigma_v
        w_v = 2 * np.sqrt(self.beta) * sigma_v

        # # compute the growing estimate of L
        # if X.shape[0] <= 3:
        #     self.L_ub = 10.0
        # else:
        #     _, unique_mask = np.unique(X, return_index = True)
        #     X = X[unique_mask]
        #     V = V[unique_mask]
        #     diff_X = np.power(np.abs(X - X.reshape(1, -1)), self.p) # NxN
        #     diff_V = np.power(np.abs(V - V.reshape(1, -1)), self.p) # NxN
        #     ind_ut = np.triu_indices(X.shape[0], 1)
        #     self.L_lb = np.max(diff_V[ind_ut] / diff_X[ind_ut])
        #     self.L_ub = self.L_lb * self.iter * self.kappa
        # #     print("the L_lb at iter {} is {}".format(self.iter, self.L_lb))

        # self.lp_dist = self.L_ub * self.diff_norm
       

        return l_f, u_f, w_f, l_v, u_v, w_v

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.

        S, M, G = self.S[-1], self.M[-1], self.G[-1]

        std_list = (M | G) * np.maximum(self.w_f, self.w_v) # only explore within the M u G set

        # # print('Length of confidence: ', S.sum() / len(S))
        # if len(self.x_ob) > 7 and self.change_time < 2 and S.mean() < 0.3 and self.n_violation == 0:
        #     # in this case, we want to explore once, to the most optimistic place
        #     self.change_time += 1
        #     print("take a chance!")
        #     np.random.seed(0)
        #     x = self.x_grid[np.argmax(self.w_v)] + 0.2 * self.step * (np.random.random() - 0.5)
        #     # x = self.x_grid[np.argmax(w_f)] + 0.2 * self.step * (np.random.random() - 0.5)
        #     # smoothed = scipy.ndimage.gaussian_filter1d(input=S, sigma=1.0, mode='constant', cval=1.0)
        #     # x = self.x_grid[np.argmin(smoothed)] + 0.2 * self.step * (np.random.random() - 0.5)
        # else:
        #     np.random.seed(0)
        #     x = self.x_grid[np.argmax(std_list)] + 0.2 * self.step * (np.random.random() - 0.5)  # add some randomness into it
        np.random.seed(0)
        x = self.x_grid[np.argmax(std_list)] + 0.2 * self.step * (np.random.random() - 0.5)  # add some randomness into it
        return np.clip(x, domain[0, 0], domain[0, 1]).reshape(1, -1)

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            # print("optimization")
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain, approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        # raise NotImplementedError

        kf_xX = self.f_kernel(x.reshape(-1, 1), np.array(self.x_ob).reshape(-1, 1))

        mean_f = self.f_mean + (kf_xX @ self.k_inv_f).reshape(-1)
        cov_f = self.kf_xx - kf_xX @ np.linalg.solve(self.kf_XX, kf_xX.T)
        sigma_f = np.sqrt(np.diagonal(cov_f))

        upper_bound = mean_f + np.sqrt(self.beta) * sigma_f

        return upper_bound[0]

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        # raise NotImplementedError

        # seems like the initial point is always safe

        self.x_ob.append(x.reshape(-1)[0])
        self.f_ob.append(f.reshape(-1)[0])
        self.v_ob.append(v.reshape(-1)[0])

        # if v.reshape(-1)[0] < self.threshold: # if v violate the safety threshold
        #     self.n_violation += 1

        # print(len(self.x_ob), x.reshape(-1)[0], f.reshape(-1)[0], v.reshape(-1)[0])

        if len(self.x_ob) == 1:
            # init value build S_0 now
            # select the nearest two/one of the grid x
            # x_left <= x <= x_right
            # left_mask = (self.x_grid <= x) * (self.x_grid + self.step > x)
            # right_mask = (self.x_grid >= x) * (self.x_grid - self.step < x)
            # S_0 = (left_mask + right_mask)
            S_0 = (self.x_grid - 0.5 * self.step <= x) * (self.x_grid + 0.5 * self.step > x)
            self.S.append(S_0)

        # update each function, S, G, M

        S_priv = self.S[-1]

        self.l_f, self.u_f, self.w_f, self.l_v, self.u_v, self.w_v = self.confidence_on_grid()

        # the v approximated at x' (axis=0) using x \in S_priv (axis=1)
        safe_v = (self.l_v.reshape(1, -1) - self.lp_dist) >= self.threshold
        # we only choose those x s.t. is in S_priv
        safe_v = safe_v * S_priv.reshape(1, -1)
        # then we union all those x'
        S_cur = (safe_v.sum(axis=1, keepdims=False) + self.S[0]) > 0  # (n_grid,), I make it union with S_0 to make sure S_cur is not empty


        # maximization area
        M_cur = S_cur * (self.u_f >= self.l_f[S_cur].max())
        

        # exploration area
        Not_S_cur = np.logical_not(S_cur)
        # G:= {x in S_cur | exists x' in D/S_cur s.t. u_v(x) - Ld(x,x') > h}
        # x set to be axis 0
        # x' set to be axis 1
        optim_save_v = (self.u_v.reshape(-1, 1) - self.lp_dist) >= self.threshold
        optim_save_v = optim_save_v * Not_S_cur.reshape(1, -1)
        G_cur = (optim_save_v.sum(axis=1, keepdims=False) * S_cur.reshape(-1)) > 0

        self.S.append(S_cur)
        self.G.append(G_cur)
        self.M.append(M_cur)

        # print(S_cur.mean(), G_cur.mean(), M_cur.mean())

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        # raise NotImplementedError

        # return x with highest u_f
        S = self.S[-1]
        return self.x_grid[np.argmax(S * self.u_f)].reshape(-1)[0]


""" Toy problem to check code works as expected """


def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return -np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()

    n_dim = 1

    # Add initial safe point
    x_init = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(1, n_dim)
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()
