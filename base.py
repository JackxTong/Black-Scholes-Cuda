import torch
import matplotlib.pyplot as plt

class GBM:
  def __init__(self, mu, sigma, s0=1., dt=0.01, device='cpu'):
      self.device = torch.device(device)

      self.mu = mu
      self.sigma = sigma
      self.s0 = s0
      self.dt = torch.tensor([dt], device=self.device)

      self.paths = None


  def simulate(self, n_paths=50, n_steps=20, return_paths=False):
      paths = torch.zeros(n_paths, n_steps+1, device=self.device)
      paths[:,0] = self.s0

      # Incremental Brownian motion
      dW = torch.randn(n_paths, n_steps, device=self.device) * torch.sqrt(self.dt)

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

    def __init__(self, mu, sigma, s0=1.0, strike = 1.0, dt=0.01, option_type = 'Call', device='cuda'):

        super().__init__(mu, sigma, dt=dt, device=device)
        if s0 < 0:
            raise ValueError("Initial asset price must be positive")
        if strike < 0:
            raise ValueError("Strike price must be positive")
        if option_type not in ['Call', 'Put']:
            raise ValueError("Option type must be either 'Call' or 'Put'")
        
        self.s0 = torch.tensor([s0], requires_grad=True)
        self.strike = torch.tensor([strike], device=self.device)
        self.type = option_type

    def price(self):
        """
        returns payoff of option
        """
        # for each path, take final price and subtract the strike price. take mean among all paths
        if self.type == 'Call':
            return torch.max(self.paths[:, -1] - self.strike, torch.zeros_like(self.paths[:, -1], device=self.device)).mean()
        else:
            return torch.max(self.strike - self.paths[:, -1], torch.zeros_like(self.paths[:, -1], device=self.device)).mean()

    def compute_Delta_and_Gamma(self):
        if self.s0.grad is not None: # Zero out any old gradient on self.s0.
            self.s0.grad.zero_()
        price_ = self.price()

        # Compute delta with create_graph=True, so that we can differentiate again
        grad_s0 = torch.autograd.grad(price_, self.s0, create_graph=True)[0]

        delta = grad_s0.item()
        gamma_ = torch.autograd.grad(grad_s0, self.s0, create_graph=False)[0]
        gamma = gamma_.item()

        print('Delta: ', delta)
        print('Gamma: ', gamma)
        return delta, gamma
    
    def compute_delta(self):
        """
        Delta is computed as the gradient of the option's price with respect to s0.
        This is an easier way to compute delta than compute_Gamma
        """
        price_ = self.price()
        price_.backward()

        return self.s0.grad.item()


def black_scholes(S, K, T, r, sigma, option_type='call', return_delta=False, return_gamma=False):
    # Ensure all inputs are tensors
    S, K, T, r, sigma = map(torch.tensor, (S, K, T, r, sigma))

    # Calculate d1 and d2
    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)

    # CDF and PDF of the standard normal distribution
    N = torch.distributions.Normal(0, 1).cdf
    phi = lambda x: torch.exp(-0.5 * x ** 2) / torch.sqrt(torch.tensor(2 * torch.pi))


    # Calculate price based on option type
    if option_type == 'call':
        price = S * N(d1) - K * torch.exp(-r * T) * N(d2)
        delta = N(d1)  # Delta for a call option
    elif option_type == 'put':
        price = K * torch.exp(-r * T) * N(-d2) - S * N(-d1)
        delta = N(d1) - 1  # Delta for a put option
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    gamma = phi(d1) / (S * sigma * torch.sqrt(T))

    if return_delta:
        return price, delta
    elif return_gamma:
        return price, delta, gamma
    return price



