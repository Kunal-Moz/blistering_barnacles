#import os
#import sys

import numpy as np
import random as rand
import matplotlib.pyplot as plt
import scipy
import math


#### parameters for simulations ####
parameter = {
    'mx': 64,  # Lattice sites in X-direction
    'my': 64,  # Lattice sites in Y-direction
    'nwarm':200000, # Warming steps
    'nskip':4096,  # Measurement Steps
    'Tc':1.0,      # Critical temperature
    'coef':0.1,    # Coeeficient for Electric Field
    'ds':1.0,      # Disorder parameter
    'dphi':0.2,    # Step for incremneting order parameter
    'gamma':0.05,  # Damping factor
    'r0':2.0,      # Stiffness of the order parameter
    'Rload':0.5    # Load Resistance
}


#### Define Class for Ginzburg Landau Free Energy ####

class Ginz_Landau_FE:
    def __init__(self,parameter,E,Tb,phi,seed):
        self.parameter = parameter
        for iparam,param in parameter.items():
            setattr(self, iparam, parameter[iparam])
        self.E = E                                 ## Electric Field added as an input
        self.Tb = Tb                               ## Bath Temperature
        self.phi = phi                             ## Order parameter lattice
        self.ms = np.ones((self.mx,self.my))       ## Mean Field order parameter (to determine Teff)
        self.v0 = np.ones((self.mx,self.my))
        self.phi_new = np.zeros((self.mx,self.my)) ## Copy of order parameter (may be commented out)
#         self.v0 = disorder*np.random.rand((mx,my))
        self.Teff = self.Tb*np.ones((self.mx,self.my)) ## Effective Temperature of the lattice
        self.dx = 1.0      ## Step along X-axis
        self.dy = 1.0      ## Step along Y-axis
        self.Lx = self.dx*(float(self.mx))  ## Length of sample
        self.Ly = self.dy*(float(self.my))  ## Breadth of sample
        self.volt = self.E*self.Ly          ## Volatge bais
        self.eq_flag = True     ## Flag for equilibriation True --> Warming , False --> Measurement
        self.seed = seed    ## Random Seed
    
    ## Compute Gradient of any 2x2 array (periodic along X, but not Y)
    def gradient(self,f):
        du = np.zeros((self.mx,self.my))
#         for i in range(self.mx):
#             for j in range(self.my):
#                 dg[i,j] = self.grad(f[i,j],i,j)
        du[1:-1,0:] += ( (f[2:,0:] - f[1:-1,0:]) /self.dx)**2 + ((f[0:-2,0:] - f[1:-1,0:])/self.dx)**2
        du[:,1:-1] += ((f[:,2:] - f[:,1:-1])/self.dy)**2 + ((f[:,0:-2] - f[:,1:-1])/self.dy)**2
        du[0,:] += ((f[1,:] - f[0,:])/self.dx)**2 + ((f[-1,:] - f[0,:])/self.dx)**2
        du[-1,:] += ((f[0,0:] - f[-1,0:])/self.dx)**2 + ((f[-2,0:] - f[-1,0:])/self.dx)**2
        du[:,0] += ((f[:,1] - f[:,0])/self.dy)**2
        du[:,-1] += ((f[:,-2] - f[:,-1])/self.dy)**2
        return du

    ## Local Gradient
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
        term1 = 0.5*((self.Teff/self.Tc) - 1.0)*self.phi**2
        term2 = 0.5*self.r0*self.gradient(self.phi)
        term3 = 0.25*self.phi**4
        fe = term1 + term2 + term3
        return fe*self.dx*self.dy
    
    ## Local Free Energy computation (Not used but may be used)
    def local_fe(self, phi_loc, Teff_loc, i, j):
        self.phi_new = self.phi_new + self.phi
        self.phi_new[i,j] = phi_loc
        del_phi = self.gradient(self.phi_new)
        fij = 0.5*((Teff_loc/self.Tc) - 1.0)*phi_loc**2 + 0.25*phi_loc**4 + 0.5*self.r0*self.grad_phi(phi_loc,i,j)
        self.phi_new = np.zeros((self.mx,self.my))
        return fij*self.dx*self.dy
    
    ## Calculating Energy difference between two points on the lattice
    def dfree(self,phi1,Teff1,i,j):
        df = -0.5*(phi1**2-self.phi[i,j]**2) + 0.5*Teff1/self.Tc*phi1**2 - 0.5*self.Teff[i,j]/self.Tc*self.phi[i,j]**2
        df += 0.25*(phi1**4-self.phi[i,j]**4)
        df += 0.5*self.r0*(self.grad_phi(phi1,i,j) - self.grad_phi(self.phi[i,j],i,j))
        return df*self.dx*self.dy

    ## Monte Carlo Loop
    def heatbath_loop(self,f):
        ## set Random Seed
        self.seed += 1
        np.random.seed(self.seed)
        ncount = 0
        iflip = 0
        icount = np.zeros((self.mx,self.my),dtype=int)
        self.ms = self.ms*0.0
        ## set flag
        if(self.eq_flag):
            nsteps = self.nwarm
        else:
            nsteps = self.nskip
            
        for ii in range(0,nsteps):
            i = np.random.randint(0,self.mx)
            j = np.random.randint(0,self.my)
            ## original order parameter
            phi0 = self.phi[i,j]
            f0 = self.local_fe(self.phi[i,j],self.Teff[i,j],i,j)
            ## new order parameter
#             phi1 = self.phi[i,j] + 2.0*self.dphi*(np.random.rand() - 0.5)
            phi1 = self.phi[i,j] + np.random.uniform(-self.dphi,self.dphi)
            Teff1 = self.Teff[i,j]
            f1 = self.local_fe(phi1,Teff1,i,j)
            ## computing the differnce in energy
            df = self.dfree(phi1,Teff1,i,j)
            icount[i,j] += 1
            ncount += 1
            ## MC MC step
            if np.random.rand() < np.exp(-df/self.Tb):
                self.phi[i,j] = phi1
                self.Teff[i,j] = Teff1
                f += df
                iflip += 1
                                
            self.ms[i,j] += self.phi[i,j]
            
        self.ms = np.where(icount == 0, self.phi , self.ms/icount)
        rate = float(iflip)/float(ncount)
        
        return self.phi, self.Teff, self.ms, f

    ## Compute Effective Temperature
    def setTeff(self):
        tbath = self.Tb*np.ones((self.mx,self.my))
        gam = self.gamma*np.ones((self.mx,self.my))
        efield = self.coef*self.E*np.ones((self.mx,self.my))
        return np.sqrt(tbath**2 + efield**2/(gam**2 + self.ms**2) )
    
    ## Compute Averages
    def averages(self,f):
        ave_phi = np.mean(self.phi)
        std_phi = np.std(self.phi)
        ave_phi2 = np.mean(self.phi**2)
        std_phi2 = np.std(self.phi**2)
        ave_fe = f/(self.mx*self.my)
        std_fe = np.sqrt((f**2 - ave_fe**2)/(self.mx*self.my))
        ave_teff = np.mean(self.Teff)
        return std_fe, std_phi, ave_phi, ave_fe, ave_teff, ave_phi2, std_phi2
    
    ## Function initiating the Warming of the lattice
    def warming(self):
        self.Teff = self.setTeff()
        f = np.sum(self.free_energy())
        self.phi, self.Teff, self.ms, f = self.heatbath_loop(f)
        return self, f
    
    ## Function intiating the production runs and gives out the measurements
    def meas(self,f):
        self.eq_flag = False
        self.Teff = self.setTeff()
        f = np.sum(self.free_energy())
        self.phi, self.Teff, self.ms, f = self.heatbath_loop(f)
        std_fe, std_phi, ave_phi, ave_fe, ave_teff, ave_phi2, std_phi2 = self.averages(f)
        ft = self.free_energy()
        return std_fe, std_phi, ave_phi, ave_fe, ave_teff, ave_phi2, std_phi2, ft.flatten(), self.phi


    
#### Equilibrium Run : Over increasing Values of Tbath for a constant value of E ####
def run_tloop():
    Mx = parameter["mx"]
    My = parameter["my"]
    
    nmeas = 8
    Tbvals = np.arange(0.05,1.2,0.05)
    ndata = np.size(Tbvals)
    print("Tbvals :",Tbvals)
    E = 0.1
    Data_Set = np.zeros((ndata,8))
    k = 0
    phit = np.ones((Mx,My))
    seed = 3245
    for Tb in Tbvals:
        glt = Ginz_Landau_FE(parameter,E,Tb,phit,seed)
        fm = glt.warming()
        Data_Set[k,0] = Tb
        for i in range(0,nmeas):
            f2 , delta2 , delta, f , teff , ohm, ohm2, ft, phit = glt.meas(fm)
            Data_Set[k,1:8] += [delta , delta2 , f, f2 , ohm, ohm2, teff]
            
        Data_Set[k,1:8] = Data_Set[k,1:8]/nmeas
        print(Data_Set[k,:])
        k += 1
        seed += 1

    np.savetxt('delta_64x64.dat',Data_Set)
       
        
      
      

if __name__ == "__main__":
    run_tloop()
    #np.savetxt('delta_20x20.dat', Data_Set)

