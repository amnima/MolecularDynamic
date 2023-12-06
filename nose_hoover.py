"""
The Nose-Hoover algorithm is non Hamiltonian but it is time reversible
Author: Amin Mirzai
"""

import numpy as np
import matplotlib.pylab as plt
import math
import sys


Avogadro = 6.02214086e23     # Avogadro number
Boltzmann = 1.3806485e-23  # Boltzmann constant

def read_input(filename):
    params = {}
    try:
        f=open(filename,"r")
        lines=f.readlines()
        # parsing the input lines
        for elements in lines:
            string_list = elements.split(':')
            key = str(string_list[0].strip())
            value = string_list[1].strip('\n')
            if key == 'nparticles':
                value = int(value)
            elif key == 'type':
                value = value
            else:
                value = float(value)

            if key == 'box':
                params[key] = (0,value), (0,value), (0,value)
            else:
                params[key] = value
        f.close
        return params
    except IOError:
        print("The file {} cannot be opened.".format(filename))
        sys.exit()

def dump_output(vels):
    with open('output_NH.txt', 'w') as f:
        f.write(str(vels))
    f.close

    



def computeForce(force, mass, vels, gamma):
    force =  force - (gamma* mass[np.newaxis].T * vels)
    return force


def updatePosVel(mass, position, vels, force, dt, gamma):
    
    position += vels * dt + ((force/mass[np.newaxis].T) - gamma * mass[np.newaxis].T *vels) * ((dt)**2/(2*mass[np.newaxis].T))
    half_vels = vels + (force - gamma * mass[np.newaxis].T * vels) * (dt/(2*mass[np.newaxis].T))
    return half_vels


def run(params):

    nparticles, box, dt, temp = params['nparticles'], params['box'], params['dt'], params['temp']
    mass, relax, nsteps, M = params['mass'], params['relax'], params['steps'], params['M']

    dims = len(box)
    mass = np.ones(nparticles) * mass / Avogadro

    vels = np.random.rand(nparticles, dims)
    position = np.random.rand(nparticles, dims)

    for i in range(dims):
        position[:,i] = box[i][0] + (box[i][1] -  box[i][0]) * position[:,i]
    #M_tilda = 3 * Boltzmann * M / nparticles

    

    #since the box is 3D we should fill the box in each dimension
    for i in range(dims):
        position[:,i] = box[i][0] + (box[i][1] -  box[i][0]) * position[:,i]

    step = 0
    gamma = 0
    result = []  
    
    gamma_half = 0
    force = 0
  

    while step <= nsteps:
        
        # force is langevin force that needs to be changed!
        force = computeForce(force, mass, vels, gamma)
        half_vels = updatePosVel(mass, position, vels, force, dt, gamma)
        
    
        for i in range(dims):
            if (position.any()<= box[i][0] or position.any()>= box[i][1]):
                vels *= -1

        # this section should be updated

        gamma_half = gamma +  (dt/2) * (np.sum(mass[np.newaxis].T * vels**2/2) - (3*nparticles + 1)* Boltzmann * temp/2)

        gamma = gamma_half + (dt/2) * (np.sum(mass[np.newaxis].T * half_vels**2/2) - (3*nparticles + 1)* Boltzmann * temp/2)
        
        vels = (half_vels + (dt/2 * force /mass[np.newaxis].T))/(1+(dt/2)* gamma)
        
        temp_inst = (2/(3*nparticles*Boltzmann))* np.sum(np.dot(mass, (vels - vels.mean(axis=0))**2))
        instant_temp = np.sum(np.dot(mass, (vels - vels.mean(axis=0))**2))/ (Boltzmann * dims * nparticles)
        print(force)
        dump_output(vels)
        

        step += 1
        result.append([dt * step, instant_temp])
        
    return np.array(result)

if __name__ == '__main__':

    #new_params = read_input('input.txt')
    params = {
    'type' : 'NH',
    'nparticles': 1000,
    'temp': 300,
    'mass': 2e-3,
    'radius': 120e-12,
    'relax': 1e-13,
    'dt': 1e-16,
    'M' : 0.001,
    'steps': 10000,
    'freq': 100,
    'box': ((0, 1e-8), (0, 1e-8), (0, 1e-8)),
    'ofname': 'traj-hydrogen-3D.dump'
    }

    result = run(params)

plt.plot(result[:,0] * 1e12, result[:,1])
plt.xlabel('time(ps)')
plt.ylabel('Temperature(K)')
plt.title('Nose-Hoover thermostat')
#plt.savefig('random.png')
plt.show()
print('The job is DONE!')

