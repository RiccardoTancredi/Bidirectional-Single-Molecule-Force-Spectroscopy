import numpy as np
from tqdm import tqdm


class Bidirectional_SMFS:
    def __init__(self, N:int=500, forward:bool=True, do_equilibrium:bool=True, reps:int=0) -> None:
        self.N = N    # particles
        self.D = 1
        self.beta = 1
        self.dt = 0.001
        self.eq_steps = 100
        self.k_s = 15
        self.low = -1.5
        self.high = 1.5
        self.forward = forward
        self.do_equilibrium = do_equilibrium
        self.tot_t = 0.75
        self.N_steps = 750
        self.reps = reps
        
        self.z_t = np.linspace(self.low, self.high, self.N_steps).tolist() + [self.high]
        if not forward:
            self.z_t = np.linspace(self.high, self.low, self.N_steps).tolist() + [self.low]
        self.z_t = np.array(self.z_t)
        self.main()


    def gen_init(self, proposal_width=0.5):
        if self.forward:
            x = 0.0
            samples = []
            t = 0

            for _ in range(self.N):
                x_new = np.random.normal(x, proposal_width)
                acceptance_ratio = np.exp(-self.beta * (self.Hamiltonian(x_new, t) - self.Hamiltonian(x, t)))
                
                if acceptance_ratio >= np.random.rand():
                    x = x_new
                samples.append(x)
            
        else:
            file_name = f'res/FORWARD_data.dat' # _{self.reps}
            samples = np.loadtxt(file_name)[-1, :]

        return np.array(samples)


    ##############
    ### Energy ###
    ##############

    def H0(self, z):
        return 5.*(z**4.) - 10.*(z**2) + 3.*z

    def V_t(self, z, t):
        return (self.k_s/2.) * (z-self.z_t[t])**2

    def Hamiltonian(self, z, t):
        return self.H0(z) + self.V_t(z, t)


    ##############
    ### Forces ###
    ##############

    def harmonic_force(self, z, t):
        return -self.k_s * (z - self.z_t[t])

    def force_H0(self, z):
        return -(20.* z**3 - 20*z + 3)


    def Velocity_Verlet(self, t):
        # Overdamped Langevin equation
        force = self.force_H0(self.z[t-1, :]) + self.harmonic_force(self.z[t-1, :], t-1)
        xi = np.random.normal(size=self.N)
        self.z[t, :] = self.z[t-1, :] +  force * self.dt + np.sqrt(2.*self.dt)*xi


    def compute_work(self, t):
        self.work[t, :] = self.work[t-1, :] + self.Hamiltonian(self.z[t], t) - self.Hamiltonian(self.z[t], t-1)


    def equilibrium(self, time=1):
        # time is fixed to 1: we are only equilibrating the system
        # the external potential is not active
        # print('Doing equilibrium steps...')
        for _ in range(self.eq_steps):
            force = self.force_H0(self.z[time-1, :]) + self.harmonic_force(self.z[time-1, :], time-1)
            xi = np.random.normal(size=self.N)
            self.z[time-1, :] += force * self.dt #  + np.sqrt(2.*self.dt)*xi


    def do_MC(self, t):
        self.Velocity_Verlet(t)  

    
    def _save_results(self):
        folder = 'res/'
        ext = '.dat'
        file_name = folder + ('FORWARD' if self.forward else 'BACKWARD')
        
        # print('Saving results...')
        np.savetxt(file_name + f'_data{ext}', self.z)           #_{self.reps}
        np.savetxt(file_name + f'_Work{ext}', self.work[1:, :]) # _{self.reps}


    def main(self):
        self.z = np.zeros(shape=(self.N_steps+1, self.N))
        self.work = np.zeros(shape=(self.N_steps+1, self.N))
        
        # print('Generating initial data...')
        self.z[0, :] = self.gen_init()

        if self.do_equilibrium:
            self.equilibrium()

        # print('Starting simulation...')
        
        for t in range(1, self.N_steps+1):
            self.do_MC(t)
            # if t == self.N_steps:
            #     self.equilibrium(time=t+1)
            #     break    
            self.compute_work(t)

        # print('Ending simulation...')

        self._save_results()



def main():
    # for reps in tqdm(range(50)):    
    for forward in [True, False]:
        # print(f'\n{"Forward" if forward else "Backward"} trajectories...\n')
        Bidirectional_SMFS(forward=forward, do_equilibrium=forward, reps=0)


if __name__== "__main__":
    main()