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

from mpi4py import MPI

np.seterr(divide='ignore', invalid='ignore')
path_to_file = "Data_psd2/vrun4_40/"
#### MPI commands (Comment out the following lines if not using MPI)
comm = MPI.COMM_WORLD  ## set up MPI
NPEs = comm.Get_size()  ## # of processors
myPE = comm.Get_rank()  ## process rank
#### parameters for simulations ####
parameter = {
    'mx': 64,       # Size of lattice
    'my': 64,       # Size of lattice
    'nwarm':400000, # Warming steps
    'nskip':2,     # Measurement steps (multiplied later by mx*my)
    'Tc':1.0,       # Some scaling parameter
    'coef':0.1,     # Coefficient for the Electric Field
    'dphi':0.2,     # Monte Carlo increment step
    'gamma':0.1,    # Damping Parameter
    'r0': 2.0,      # Strength of Coupling between adjacent sites
    'g2' : 1.0,     # Coefficient of 2nd order term in Ginzburg-Landau Free Energy
    'g4' : 1.0,    # Coefficient of 4th order term in GL Free Energy
    'g6' : 0.0,     # Coefficient of 6th order term in GL Free Energy
    'Rload':1.0,    # Load Resistance
    'mfphi': False, # If measurements are mean field or not
    'tloop': False  # True --> E : constant, Temperature sweep , False --> T_bath and Applied Voltage upsweep
}

#### Run parameters ####
run_par = {
    ### for voltage loop
    'Tcrit':0.55,
    'dE' : 0.2,      # Increment steps for Electric Field
    'minE' : 0.2,    # Min value of Electric Field
    'maxE' : 15.2,    # Max value of Electric Field
    ### for temperature loop
    'dTb' : 0.01,
    'minT' : 0.01,
    'maxT' : 0.81,
    'nmeas' : 16384,   # Number of measurements
    'Tbath' : 0.40,   # Bath temperature
    'seed' : 5823588,  # Random number generator seed
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
        self.dE = E                    ## Electric Field added as an input
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
        self.volt = self.dE*self.Ly         ## Volatge bais
        self.Ex = np.zeros((self.mx,self.my)) ## Electric Field along X-direction in the lattice
        self.Ey = np.zeros((self.mx,self.my)) ## Electric Field along the Y-direction in the lattice
#        self.resist = np.zeros((self.mx,self.my)) ## Resistance network
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
        # fe = 0.0
        term1 = (1/2.)*self.g2*((self.Teff/self.Tc) - 1.0)*self.phi**2
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
        # np.random.seed(self.seed + np.random.randint(-99999,99999))
        # self.seed += 4
        # idummy = np.random.randint(100000, 10000000)
        idummy = self.seed
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
        #hb_loop(phi,Teff,f,ms,mfphi,icount,rate,nstep,dx,dy,idummy,Tc,Tbath,dphi,r0,v0,gamma,g2,g4,g6,mx,my)
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
#            print("yes")
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
        temp = np.zeros(10)
        temp[0] = np.mean(self.phi)   ## Average of phi values
        temp[1] = f/(self.mx*self.my)  ## Average free energy per site
        temp[2] = np.mean(self.Teff)   ## Average Temperature of the system
        temp[3] = (self.volt - self.pot[self.mtot])  ## Sample Voltage
        temp[4] = (self.volt - self.pot[self.mtot])**2  ## Variation in the Voltage
        temp[5] = self.pot[self.mtot]/(self.Rload)  ## Current 
        temp[6] = (self.volt - self.pot[self.mtot])/(self.pot[self.mtot]/self.Rload) ## Resistance of the Sample
        temp[7] = ((self.volt - self.pot[self.mtot])/(self.pot[self.mtot]/self.Rload))**2  ## Varaition in Resistance
        temp[8] = (self.volt - self.pot[self.mtot])/self.my ## Sample Electtric Field
        temp[9] = self.pot[self.mtot]/(self.Rload*self.mx) ## Current Density
        return temp
    
    ## Function initiating the Warming of the lattice
    def warming(self):
        ## Kirchhoff function runs from Fortran Routine : Kirchhoff
        if self.mfphi:
            self.pot = kirchhoff.kirchhoff(self.gamma, self.ms, self.Rload, self.volt, self.pot, self.mtot, self.mx, self.my)
        else:
            self.pot = kirchhoff.kirchhoff(self.gamma, self.phi, self.Rload, self.volt, self.pot, self.mtot, self.mx, self.my)
#        print("potential :", self.pot)
        self.Ex, self.Ey = self.Efield()
        self.Teff = self.setTeff()
        f = np.sum(self.free_energy())
        self.phi, self.Teff, self.ms, f = self.heatbath_loop(f)
        return self, f
    
    ## Function intiating the production runs and gives out the measurements
    def meas(self,f):
        self.eq_flag = False
        ## Kirchhoff function runs from Fortran Routine : Kirchhoff
        if self.mfphi:
            self.pot = kirchhoff.kirchhoff(self.gamma, self.ms, self.Rload, self.volt, self.pot, self.mtot, self.mx, self.my)
        else:
            self.pot = kirchhoff.kirchhoff(self.gamma, self.phi, self.Rload, self.volt, self.pot, self.mtot, self.mx, self.my)
#        print("potential :", self.pot)
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
    nmeas = nmeas*NPEs
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
        Tb = run_par['Tbath']*run_par['Tcrit']   ## Bath Temperature
        Evals = np.arange(minE, maxE, dE)    ## Set
        # Evals = np.append(Evals, Evals[::-1])
        ndata = np.size(Evals)
        print( "Non-Equilibrium Run")
        print("Electric Field Values :" , Evals)

    Data_Set = np.zeros((ndata, 11))
    phit = np.ones((Mx, My))
    # phit = np.loadtxt(path_to_file + "")
    msp = np.ones((Mx, My))
    seed = run_par["seed"] - myPE
    for k in range(ndata):
        if tloop:
            glt = Ginzburg_Landau_FE(parameter, E, Tbvals[k], phit, msp, seed)
            Data_Set[k, 0] = Tbvals[k]
        else:
            glt = Ginzburg_Landau_FE(parameter, Evals[k], Tb, phit, msp, seed)
            Data_Set[k, 0] = glt.volt
        fm = glt.warming()
        
        temp_data_tt = np.zeros((nmeas//NPEs, 10))
        temp_data = np.zeros((nmeas, 10))
        # temp_data_tt = comm.scatter(temp_data[(nmeas//NPEs)*myPE:(nmeas//NPEs)*(myPE+1),:], root=0)
        for i in range(0, nmeas//NPEs):
            temp_data_tt[i, :] = glt.meas(fm)
            # if myPE == 0:
                # if i == ((nmeas//NPEs)-1):
                #     np.savetxt(path_to_file + "phi_val_" + format(Evals[k], '.2f') +".dat", glt.phi)
        comm.Gather(temp_data_tt ,temp_data, root = 0) #[nmeas*myPE:nmeas*(myPE+1),:]
        if myPE == 0:
            # if Tbvals[k] == 1.2:
            #     np.savetxt(path_to_file + "phi_val_" + format(Tbvals[k], '.2f') +".dat", glt.phi)
            np.savetxt(path_to_file + "t_data_" + str(k) + ".dat",temp_data[:, 6])
        Data_Set[k, 1:11] = np.mean(temp_data, axis=0)
        ft = glt.free_energy()
        phit = glt.phi
        msp = glt.ms
    if myPE == 0:
        np.savetxt(path_to_file + "delta_f2py_test.dat", Data_Set)
    time1 = time.time() - start
    print(time1)
      

if __name__ == "__main__":
    run_mcloop()
    
           

