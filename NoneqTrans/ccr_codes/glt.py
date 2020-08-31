import os
os.system("python -m numpy.f2py -c Montecarlo.f90 -m montecarlo")
os.system("python -m numpy.f2py -c Kirchoff_sub.f90 -m kirchhoff")

import numpy as np
import random as rand
import matplotlib.pyplot as plt
import math
import time
import montecarlo as mcmc
import kirchhoff
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as spla
import scipy.linalg.lapack as la

np.seterr(divide='ignore', invalid='ignore')
path_to_file = "Data/vrun24_sd/"

#### parameters for simulations ####
parameter = {
    'mx': 64,       # Size of lattice
    'my': 64,       # Size of lattice
    'nwarm':200000, # Warming steps
    'nskip':16,     # Measurement steps (multiplied later by mx*my)
    'Tc':1.313,       # Critical Temperature (Scaled Value)
    'coef':0.1,     # Coefficient for the Electric Field
    'dphi':0.2,     # Monte Carlo increment step
    'gamma':0.1,   # Damping Parameter
    'r0': 3.5,       # Strength of Coupling between adjacent sites
    'g2' : 1.0,     # Coefficient of 2nd order term in Ginzburg-Landau Free Energy
    'g4' : -1.0,    # Coefficient of 4th order term in GL Free Energy
    'g6' : 0.5,     # Coefficient of 6th order term in GL Free Energy
    'Rload':0.5,    # Load Resistance
    'mfphi':False,   # If measurements are mean field or not
    'tloop': False  # True --> T_bath varies and E is constant , False --> T_bath constant and E is varying
}

#### Run parameters ####
run_par = {
    ### for voltage loop
    'dE' : 0.1,      # Increment steps for Electric Field
    'minE' : 0.1,    # Min value of Electric Field
    'maxE' : 32.1,    # Max value of Electric Field
    ### for temperature loop
    'dTb' : 0.01,
    'minT' : 0.01,
    'maxT' : 1.61,
    'nmeas' : 1024,   # Number of measurements
    'Tbath' : 0.85,   # Bath temperature
    'seed' : 885253,  # Random number generator seed
    'NPEs' : 1       # Number of Processors for MPI ( = 1 if MPI is not used )
}

fl = open(path_to_file + "input_par.dat","w")   # Storing input and run parameters
fl.write( str(parameter) )
fl.write( str(run_par) )
fl.close()

##### Environment which contains all the functions required for the simulation #####
class Ginzburg_Landau_FE:
    def __init__(self,parameter,E,Tb,phi,ms,seed):
        ## First step is loading all the input parameters as Class members
        self.parameter = parameter
        for iparam,param in parameter.items():
            setattr(self, iparam, parameter[iparam])
        self.nskip = self.nskip*self.mx*self.my   ## measurement steps
        self.mtot = self.mx*self.my    ## total number of sites
        self.E = E                    ## Electric Field added as an input
        self.Tb = Tb                    ## Bath Temperature
        self.phi = phi                  ## Order parameter lattice -- Gap
        self.ms = ms                    ## Mean field order parameter
        self.v0 = np.zeros((self.mx,self.my))  ## Order parameter
#         self.phi_new = np.zeros((self.mx,self.my)) ## Copy of order parameter (may be commented out)
#         self.v0 = disorder*np.random.rand((mx,my))
        self.Teff = self.Tb*np.ones((self.mx,self.my)) ## Effective Temperature of the lattice
        self.dx = 1.0      ## Step along X-axis
        self.dy = 1.0      ## Step along Y-axis
        self.Lx = self.dx*(float(self.mx)-1.0)  ## Length of sample
        self.Ly = self.dy*(float(self.my)-1.0)  ## Breadth of sample
        self.volt = self.E*self.Ly          ## Volatge bais
        self.Ex = np.zeros((self.mx,self.my)) ## Electric Field along X-direction in the lattice
        self.Ey = np.zeros((self.mx,self.my)) ## Electric Field along the Y-direction in the lattice
        self.resist = np.zeros((self.mx,self.my)) ## Resistance network
        self.pot = np.zeros(self.mtot+1)  ## Potential at each site
        self.eq_flag = True     ## Flag for equilibriation True --> Warming , False --> Measurement
        self.seed = seed    ## Random Seed
    
    ## Compute Gradient of any 2x2 array (periodic along X, but not Y)
    def gradient(self,f):
        du = np.zeros((self.mx,self.my))
        du[1:-1,0:] += ( (f[2:,0:] - f[1:-1,0:]) /self.dx)**2 + ((f[0:-2,0:] - f[1:-1,0:])/self.dx)**2
        du[:,1:-1] += ((f[:,2:] - f[:,1:-1])/self.dy)**2 + ((f[:,0:-2] - f[:,1:-1])/self.dy)**2
        du[0,:] += ((f[1,:] - f[0,:])/self.dx)**2 + ((f[-1,:] - f[0,:])/self.dx)**2
        du[-1,:] += ((f[0,0:] - f[-1,0:])/self.dx)**2 + ((f[-2,0:] - f[-1,0:])/self.dx)**2
        du[:,0] += ((f[:,1] - f[:,0])/self.dy)**2
        du[:,-1] += ((f[:,-2] - f[:,-1])/self.dy)**2
        return du

    ## Local Gradient (not used)
    def grad_phi(self,phi_loc,i,j):
        i1 = (i-1+self.mx)%self.mx
        i2 = (i+1)%self.mx
        dg = ((self.phi[i2,j] - phi_loc)/self.dx)**2 + ((self.phi[i1,j]-phi_loc)/self.dx)**2
        if j > 0 :
            dg += ((self.phi[i,j-1]-phi_loc)/self.dy)**2
        if j < (self.my-1):
            dg += ((self.phi[i,j+1]-phi_loc)/self.dy)**2
        return dg

    ## Compute Free Energy as an array
    def free_energy(self):
        term1 = (1/.2)*self.g2*((self.Teff/self.Tc) - 1.0)*self.phi**2
        term2 = 0.5*self.r0*self.gradient(self.phi)
        term3 = (1/4.)*self.g4*self.phi**4
        term4 = (1/6.)*self.g6*self.phi**6
        fe = term1 + term2 + term3 + term4
        return fe*self.dx*self.dy
    
    
    ## Calculating Energy difference between two points on the lattice (Not Used)
    def dfree(self,phi1,Teff1,i,j):
        df = -(1/2.)*(phi1**2-self.phi[i,j]**2) + (1/4.)*(phi1**4-self.phi[i,j]**4)
        df += (1/2.)*self.r0*(self.grad_phi(phi1,i,j) - self.grad_phi(self.phi[i,j],i,j))
        df += (1/2.)*Teff1/self.Tc*phi1**2 - (1/2.)*self.Teff[i,j]/self.Tc*self.phi[i,j]**2
        df += (1/6.)*(phi1**6-self.phi[i,j]**6)
        return df*self.dx*self.dy

    ## Monte Carlo Loop
    def heatbath_loop(self,f):
        ## set Random Seed
        np.random.seed(self.seed + np.random.randint(-99999,99999))
        self.seed += 4
        idummy = np.random.randint(100000, 10000000)
        ncount = 0
        iflip = 0
        icount = np.zeros((self.mx,self.my),dtype=int)
        self.ms = self.ms*0.0

        ## set flag
        if(self.eq_flag):
            nstep = self.nwarm
        else:
            nstep = self.nskip
        rate = 0
        ## Monte Carlo loop : using fortran code Montecarlo.f90
        self.phi, self.Teff, f, self.ms, icount, rate = mcmc.hb_loop(self.phi, self.Teff, f, self.ms,self.mfphi,
                                                                     icount, nstep, self.dx, self.dy, idummy,
                                                                     self.Tc, self.Tb, self.dphi, self.r0,
                                                                     self.v0, self.gamma, self.g2, self.g4, self.g6,
                                                                     self.mx,self.my)
        
        self.ms = np.where(icount == 0, self.phi , self.ms/icount)

        return self.phi, self.Teff, self.ms, f

    ## Compute Effective Temperature
    def setTeff(self):
        tbath = self.Tb*np.ones((self.mx,self.my))
        gam = self.gamma*np.ones((self.mx,self.my))
        efield = np.sqrt(self.Ex**2 + self.Ey**2)
        if self.mfphi:
            T1 = self.coef*efield/np.sqrt(gam**2+self.ms**2)
        else:
            T1 = self.coef*efield/np.sqrt(gam**2+self.phi**2)
        if self.tloop:
            teff = tbath
        else :
            teff = np.sqrt(tbath ** 2 + T1 ** 2)
        return teff


    ## Compute Electric Field
    def Efield(self):
        f = np.reshape(self.pot[0:self.mtot],(self.mx,self.my))
        dux = np.zeros((self.mx,self.my))
        duy = np.zeros((self.mx,self.my))
        duy[1:-1,:] = 0.5*(f[2:,:] - f[0:-2,:])/self.dy
        duy[0,:] = (f[1,:] - f[0,:])/self.dy
        duy[-1,:] = (f[-1,:] - f[-2,:])/self.dy
        duy = np.transpose(duy)
        dux[:,1:-1] = 0.5*(f[:,2:] - f[:,:-2])/self.dx
        dux[:,0] = 0.5*(f[:,1] - f[:,-1])/self.dx
        dux[:,-1] = 0.5*(-f[:,-2] + f[:,0] )/self.dx
        dux = np.transpose(dux)
        return dux,duy
    
    ## Compute local Averages
    def averages(self,f):
        temp = np.zeros(8)
        temp[0] = np.mean(self.phi)   ## Average of phi values
        temp[1] = f/(self.mx*self.my)  ## Average free energy per site
        temp[2] = np.mean(self.Teff)   ## Average Temperature of the system
        temp[3] = (self.volt - self.pot[self.mtot])/self.my  ## Sample Voltage
        temp[4] = (self.volt - self.pot[self.mtot])**2  ## Variation in the Voltage
        temp[5] = (self.pot[self.mtot]/self.Rload)/self.mx  ## Current Density in the Sample
        temp[6] = (self.volt - self.pot[self.mtot])/(self.pot[self.mtot]/self.Rload) ## Resistance of the Sample
        temp[7] = ((self.volt - self.pot[self.mtot])/(self.pot[self.mtot]/self.Rload))**2  ## Varaition in Resistance
        return temp
    
    ## Function initiating the Warming of the lattice
    def warming(self):
        ## Kirchhoff function runs from Fortran Routine : Kirchhoff
        self.pot = kirchhoff.kirchhoff(self.gamma, self.phi, self.Rload, self.volt, self.mtot, self.mx, self.my)
        self.Ex, self.Ey = self.Efield()
        self.Teff = self.setTeff()
        f = np.sum(self.free_energy())
        self.phi, self.Teff, self.ms, f = self.heatbath_loop(f)
        return self, f
    
    ## Function intiating the production runs and gives out the measurements
    def meas(self,f):
        self.eq_flag = False
        ## Kirchhoff function runs from Fortran Routine : Kirchhoff
        self.pot = kirchhoff.kirchhoff(self.gamma, self.phi, self.Rload, self.volt, self.mtot, self.mx, self.my)
        self.Ex, self.Ey = self.Efield()
        self.Teff = self.setTeff()
        f = np.sum(self.free_energy())
        self.phi, self.Teff, self.ms, f = self.heatbath_loop(f)
        temp = self.averages(f)
        return temp

        
    
    
#### Equilibrium Run : Over increasing Values of Tbath for a constant value of E ####
def run_mcloop():
    start = time.time()
    ## Next 4 lines assigns the parameter values to variables outside the Class
    nmeas = run_par['nmeas']
    Mx = parameter['mx']
    My = parameter['my']
    tloop = parameter['tloop']
    print(parameter)
    print(run_par)
    if tloop:  ## If tloop is TRUE, then we are varying the temperature by keeping the Field value constant
        E = 0.001    ## Electric Field
        dTb = run_par['dTb']
        minT = run_par['minT']
        maxT = run_par['maxT']
        Tbvals = np.arange(minT,maxT,dTb)   ## Set range of temperature :: (min Tbath, max Tbath, step ) or add the values in run_par
        # Tbvals = np.append(Tbvals, Tbvals[::-1])
        ndata = np.size(Tbvals)
        print( "Equilibrium Run")
        print("T_bath valuess :", Tbvals)
    else:   ## If tloop is FALSE, then we are varying the Electric Field by keeping the Temperature value constant
        dE = run_par['dE']  ## Increment of Electric Field
        minE = run_par['minE']  ## Min. value of E
        maxE = run_par['maxE']  ## Max value of E
        Tb = run_par['Tbath']*parameter['Tc']   ## Bath Temperature
        # Evals = np.arange(minE, maxE, dE)    ## Set
        E_1 = np.arange(minE, 20.0, dE)
        E_2 = np.arange(20.0, 25.0, 0.05)
        E_3 = np.arange(25.0, maxE, dE)
        Evals = np.concatenate((E_1, E_2, E_3))
        # Evals = np.append(Evals, Evals[::-1])
        ndata = np.size(Evals)
        print( " Non-Equilibrium Run ")
        print("E values :" , Evals)

    Data_Set = np.zeros((ndata, 10))
    phit = np.ones((Mx, My))
    msp = np.ones((Mx, My))
    seed = run_par["seed"]
    for k in range(ndata):
        if tloop:
            glt = Ginzburg_Landau_FE(parameter, E, Tbvals[k], phit, msp, seed)
            Data_Set[k, 0] = Tbvals[k]
        else:
            glt = Ginzburg_Landau_FE(parameter, Evals[k], Tb, phit, msp, seed)
            Data_Set[k, 0] = Evals[k]
        fm = glt.warming()
        temp_data = np.zeros((nmeas, 8))
        for i in range(0, nmeas):
            temp_data[i, :] = glt.meas(fm)
            if i%16 == 0 :
                if (glt.E >= 20.0) and (glt.E <= 25.0 ):
            #     # if (glt.Tb >= 1.0) and (glt.Tb <= 1.5):
                    if tloop:
                        np.savetxt(path_to_file + "phi_val_" + format(Tbvals[k], '.2f') + "_" + str(i) + ".dat", glt.phi)
                    else:
                        np.savetxt(path_to_file + "phi_val_" + format(Evals[k], '.2f') + "_" + str(i) + ".dat",glt.phi)
        np.savetxt(path_to_file + "t_data_" + str(k) + ".dat",temp_data[:, 6])
        Data_Set[k, 1:9] = np.mean(temp_data, axis=0)
        Data_Set[k, 9] = np.sqrt(Data_Set[k, 8] - Data_Set[k, 7]** 2)
        # np.savetxt("Data/run6/phi_val_" + str(k) + ".dat", glt.phi)
        ft = glt.free_energy()
        phit = glt.phi
        msp = glt.ms

    np.savetxt(path_to_file + "delta_f2py_test.dat", Data_Set)
    time1 = time.time() - start
    print(time1)
      

if __name__ == "__main__":
    run_mcloop()
    
           

