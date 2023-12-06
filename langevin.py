"""
Amin Mirzai
The following file is implementation of Langevin thermostat
REFERENCE: Modelling Materials: Continuum, Atomistic and Multiscale Techniques
Chapter: 9
"""

import numpy as np
import matplotlib.pylab as plt
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
            string_list = elements.split(',')
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


def dump_output(nparticles, nsteps, box, pos, vels, radius, mass):
    with open('output_langevin.dump', 'w') as f:
        f.write('ITEM: TIMESTEP\n')
        f.write('{}\n'.format(nsteps))

        f.write('ITEM: NUMBER OF ATOMS\n')
        f.write('{}\n'.format(nparticles))

        f.write('ITEM: BOX BOUNDS f f f\n')

        for boundaries in box:
            f.write('{} {}\t'.format(*boundaries)+'\n')

        f.write('ITEM: ATOMS' + ' v_x v_y v_z' + ' radius' + ' x y z' + ' mass' + '\n')
        for i,item in enumerate(vels):
            
            for position in item:        
                f.write(('{}' .format(position))+ '\t')
            f.write('{}'.format(radius) + '\t') 
            for j, element in enumerate(item):    
                f.write('{}'.format(pos[i,j])+'\t')
            
            f.write('{}'.format(mass) + '\t')
            f.write('\n')
                
        f.close()
    


def langevin(params):
    nparticles, radius, box, dt, temp = params['nparticles'], params['radius'], params['box'], params['dt'], params['temp']
    mass, relax, nsteps, dump_frequency = params['mass'], params['relax'], params['steps'], params['dump_freq']

    # the mass is on molar unit, so we devide by Avogadro number to convert it to the SI unit
    dim = len(box)
    pos = np.random.rand(nparticles,dim)

    for i in range(dim):
        pos[:,i] = box[i][0] + (box[i][1] -  box[i][0]) * pos[:,i]

    vels = np.random.rand(nparticles,dim)

    # mass for Neon
    mass = np.ones(nparticles) * mass *10

    step = 0
    result = []
 

    while step <= nsteps:
        step += 1
      

        """ Calculate the force """
        sigma = np.sqrt(2.0 * mass * temp * Boltzmann/ (relax * dt))
        #random forces to contribute on behalf of a heat bath
        random_force = np.random.randn(nparticles, dim) * sigma[np.newaxis].T
        # equation 9.26 in page 512 of Modelling Materials
        force = -(vels * mass[np.newaxis].T) / relax + random_force 
        pos += vels * dt
        vels += force * dt/ mass[np.newaxis].T
       
        """        
         Make sure the particle remains within boundaries of the designated box
         identify the length of the boundary
        """
        
        for i in range(dim):
            if (pos.any()<= box[i][0] or pos.any()>= box[i][1]):
                vels = vels * -1
            
       
        # instantantious temperature of the system
        instant_temp = np.sum(np.dot(mass, (vels - vels.mean(axis=0))**2))/ (Boltzmann * dim * nparticles)
      
        result.append([dt * step, instant_temp])

        if not step % dump_frequency:
            dump_output(nparticles, nsteps, box, pos, vels, radius, 10 * params['mass'])

    return np.array(result)


if __name__ == '__main__':
    params = read_input('input.txt')
    result = langevin(params)

plt.plot(result[:,0] * 1e12, result[:,1])
plt.xlabel('time(ps)')
plt.ylabel('Temperature(kelvin)')
plt.title('The Langevin thermostat')
#plt.savefig('random.png')
plt.show()
print('The job is DONE!')




