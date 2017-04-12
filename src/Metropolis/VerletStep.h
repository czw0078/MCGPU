/**
 * VerletStep.h
 *
 * A subclass of SimulationStep that uses a "verlet list" for energy
 * calculations
 *
 */

#ifndef METROPOLIS_VERLETSTEP_H
#define METROPOLIS_VERLETSTEP_H

#include "SimulationStep.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class VerletStep: public SimulationStep {
    private:
        int VERLET_SIZE;
        int VACOORDS_SIZE;
        thrust::host_vector<int> h_verletList;
        thrust::host_vector<Real> h_verletAtomCoords;
        thrust::device_vector<int> d_verletList;
        thrust::device_vector<Real> d_verletAtomCoords;

        void resizeThrustVectors();

    public:
        explicit VerletStep(SimBox* box): SimulationStep(box),
                                    h_verletList(0),
                                    h_verletAtomCoords(0),
                                    d_verletList(0),
                                    d_verletAtomCoords(0) {

            VERLET_SIZE = box->numMolecules * box->numMolecules;
            VACOORDS_SIZE = NUM_DIMENSIONS * box->numAtoms;
                                    }

        virtual ~VerletStep();
        virtual Real calcSystemEnergy(Real &subLJ, Real &subCharge, int numMolecules);

       /**
        * Determines the energy contribution of a particular molecule.
        * @param currMol The index of the molecule to calculate the contribution
        * @param startMol The index of the molecule to begin searching from to 
        *                      determine interaction energies
        * @param verletList The host_vector<int> containing the indexes of molecules
        *                      in range for each molecule 
        * @return The total energy of the box (discounts initial lj / charge energy)
        */
        virtual Real calcMolecularEnergyContribution(int currMol, int startMol);
        virtual void changeMolecule(int molIdx, SimBox *box);
        virtual void rollback(int molIdx, SimBox *box);

};

/**
 * VerletCalcs namespace
 *
 * Contains logic for calculations used by the VerletStep class.
 * Although logically related to the VerletStep class, these need
 * to be seperated to accurately run on the GPU.
 */
namespace VerletCalcs {

    /**
     * Determines the energy contribution of a particular molecule.
     * @param currMol The index of the molecule to calculate the contribution
     * @param startMol The index of the molecule to begin searching from to 
     *                      determine interaction energies
     * @param sb The SimBox on either the CPU or GPU
     * @param verletList The int* casted host_vector<int> device_vector<int> containing 
     *                      the indexes of molecules in range for each molecule 
     * @return The total energy of the box (discounts initial lj / charge energy)
     */
    __host__ __device__
    void calcMolecularEnergyContribution(int currMol, int startMol, SimBox* sb, int* verletList, int verletListLength);

    __global__ 
    void energyContribution_Kernel(int currMol, int startMol, SimBox* sb, int* verletList, int verletListLength);
     /**
      * Determines whether or not two molecule's primaryIndexes are
      * within the cutoff range of one another and calculates the 
      * energy between them (if within range)
      *
      * @param m1 Molecule 1 
      * @param m2 Molecule 2
      * @param sb The SimBox from which data is to be used 
      * @return The total energy between two molecules 
      */
    __host__ __device__
    Real calcMoleculeInteractionEnergy (int m1, int m2, SimBox* sb);

    /**
     *
     */
    __host__ __device__
    void createVerlet(int* verletList, Real* verletAtomCoords, int verletListLength, int vaCoordsLength, SimBox* sb);
    
    /**
     * Checks if the verlet list needs to be updated to account for 
     * changes to molecule positions
     *
     * @return True/False if an update needs to take place
     */
    __host__ __device__
    void updateVerlet(Real* vaCoords, SimBox* sb, int i);

    __global__
    void updateVerlet_Kernel();

    /**
     * Frees memory used by verlet lists and coordinate lists
     */
    void freeMemory(thrust::host_vector<int> &h_verletList, thrust::host_vector<Real> &h_vaCoords);
    void freeMemory(thrust::device_vector<int> &d_verletList, thrust::device_vector<Real> &d_vaCoords);

    /**
     * Creates a new verlet list for CPU run
     *
     * @return An int* representing a verlet list
     */
    __host__ __device__
    int* newVerletList(SimBox* sb, int verletListLength);

    __global__
    void newVerletList_Kernel(int* &verletList, int verletListLength);
}

#endif
