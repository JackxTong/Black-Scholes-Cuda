import torch
import matplotlib.pyplot as plt

class GBM:
    def __init__(self, mu, sigma, s0=1., dt=0.01, T = 1.0, n_paths=1000, device='cpu'):
        self.device = torch.device(device)
        self.mu = torch.tensor([mu], device=self.device)
        self.sigma = sigma
        self.s0 = s0
        self.dt = torch.tensor([dt], device=self.device)
        self.T = T
        self.n_paths = n_paths
        self.paths = None


    def simulate(self):
        try:
            n_steps = int(self.T / self.dt)
        except:
            raise ValueError("T/dt must be an integer")
        paths = torch.zeros(self.n_paths, n_steps+1, device=self.device)
        paths[:,0] = self.s0

        # Incremental Brownian motion
        dW = torch.randn(self.n_paths, n_steps, device=self.device) * torch.sqrt(self.dt)

        for i in range(1, n_steps+1):
            paths[:,i] = paths[:,i-1] + self.mu * paths[:,i-1] * self.dt + \
                            self.sigma * paths[:,i-1] * dW[:,i-1]

        self.paths = paths
        print('Simulation done')


    def to(self, device):
        self.device = torch.device(device)
        self.dt = self.dt.to(device=device)

        if self.paths is not None:
            self.paths = self.paths.to(device=device)


    def plot(self):
        if self.paths is None:
            raise RuntimeError("Paths have to be simulated first")

        plt.figure(figsize=(10,6), constrained_layout=True)
        plt.plot(self.paths.cpu().numpy().T)
        plt.grid()
        plt.title('Geometric Brownian motion with mu={}, sigma={}'.format(self.mu, self.sigma))
        plt.show()

class Option(GBM):

    def __init__(self, mu, sigma, s0=1.0, strike = 1.0, dt=0.01, n_paths=1000, option_type = 'Call', device='cuda'):

        super().__init__(mu, sigma, dt=dt, device=device, n_paths=n_paths)
        if s0 < 0:
            raise ValueError("Initial asset price must be positive")
        if strike < 0:
            raise ValueError("Strike price must be positive")
        if option_type not in ['Call', 'Put']:
            raise ValueError("Option type must be either 'Call' or 'Put'")
        
        self.s0 = torch.tensor([s0], requires_grad=True)
        self.strike = torch.tensor([strike], device=self.device)
        self.type = option_type

    def simulate(self):
        return super().simulate()

    def plot(self):
        '''Add detach() to self.paths (super() does not support.'''
        if self.paths is None:
            raise RuntimeError("Paths have to be simulated first")
        if self.n_paths > 10000:
            print("Too many paths to plot. Plotting first 10000 paths")
            plt.figure(figsize=(10,6), constrained_layout=True)
            plt.plot(self.paths[:10000].detach().cpu().numpy().T)
            plt.grid()
            plt.title('Geometric Brownian motion with mu={}, sigma={}'.format(self.mu, self.sigma))
            plt.show()
        else:
            plt.figure(figsize=(10,6), constrained_layout=True)
            plt.plot(self.paths.detach().cpu().numpy().T)
            plt.grid()
            plt.title('Geometric Brownian motion with mu={}, sigma={}'.format(self.mu, self.sigma))
            plt.show()

    def compute_payoff(self):
        """
        returns payoff of option
        """
        # for each path, take final price and subtract the strike price. take mean among all paths
        if self.type == 'Call':
            payoff = torch.max(self.paths[:, -1] - self.strike, torch.zeros_like(self.paths[:, -1], device=self.device)).mean()
        else:
            payoff = torch.max(self.strike - self.paths[:, -1], torch.zeros_like(self.paths[:, -1], device=self.device)).mean()
        self.payoff = payoff.item()
        self.price = self.payoff * torch.exp(-self.mu * self.T).detach().item()

        return payoff
    
    def compute_delta(self):
        """
        Delta is computed as the gradient of the option's price with respect to s0.
        This is an easier way to compute delta than compute_Gamma
        """
        payoff_ = self.compute_payoff()
        payoff_.backward()

        return self.s0.grad.item()

    def black_scholes(self, print_=False, return_delta=False):
        S, K, T, r, sigma = self.s0, self.strike, self.T, self.mu, self.sigma
        # Ensure all inputs are tensors
        S, K, T, r, sigma = [
            x.clone().detach().to(self.device, dtype=torch.float32) if isinstance(x, torch.Tensor) 
            else torch.tensor(x, device=self.device, dtype=torch.float32)
            for x in (S, K, T, r, sigma)
        ]

        option_type = self.type

        # Calculate d1 and d2
        d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
        d2 = d1 - sigma * torch.sqrt(T)

        # CDF and PDF of the standard normal distribution
        N = torch.distributions.Normal(0, 1).cdf
        phi = lambda x: torch.exp(-0.5 * x ** 2) / torch.sqrt(torch.tensor(2 * torch.pi))

        if option_type == 'Call':
            price = S * N(d1) - K * torch.exp(-r * T) * N(d2)
            delta = N(d1)
        elif option_type == 'Put':
            price = K * torch.exp(-r * T) * N(-d2) - S * N(-d1)
            delta = N(d1) - 1  # Delta for a put option
        else:
            raise ValueError("option_type must be 'Call' or 'Put'")
        
        gamma = phi(d1) / (S * sigma * torch.sqrt(T))
        if print_:
            print('price: ', price.item())
            print('Delta: ', delta.item())
            print('Gamma: ', gamma.item())
        self.BS_price = price.item()
        if return_delta:
            return delta.item()
        return price.item()

    def plot_accuracy(self, n_paths_values):
        """
        Plot the accuracy of the option price with the Black-Scholes price
        by varying the number of paths in the simulation.
        """
        # Define the range of n_paths to test
        errors = []
        bs_price = self.black_scholes()
        for n_paths in n_paths_values:
            self.n_paths = n_paths
            self.simulate()
            self.compute_payoff()
            mc_price = self.price

            # Compute the relative error
            error = abs(mc_price - bs_price) / bs_price
            errors.append(error)

        # Plot the errors
        plt.figure(figsize=(8, 4))
        plt.plot(n_paths_values, errors, marker='o', linestyle='-')
        plt.xscale('log')  # Logarithmic scale for the number of paths
        plt.grid(True)
        plt.xlabel('Number of Paths (log scale)')
        plt.ylabel('Relative Error')
        plt.title(f'Convergence of MC Price to BS Price for {self.type} option')
        plt.show()

    def plot_delta(self, n_paths_values):
        """
        Plot the accuracy of the option's delta with the Black-Scholes delta
        by varying the number of paths in the simulation.
        """
        errors = []

        delta_bs = self.black_scholes(return_delta=True)
        print('BS Delta: ', delta_bs)
        for n_paths in n_paths_values:
            if self.s0.grad is not None:
                self.s0.grad.zero_() # zero the gradient
            self.n_paths = n_paths
            self.simulate()
            delta_mc = self.compute_delta()
            print('MC Delta: ', delta_mc)

            # Compute the relative error
            error = abs(delta_mc - delta_bs) / delta_bs
            errors.append(error)

        # Plot the errors
        plt.figure(figsize=(8, 4))
        plt.plot(n_paths_values, errors, marker='o', linestyle='-')
        plt.xscale('log')
