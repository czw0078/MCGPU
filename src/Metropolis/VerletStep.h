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
    public:
        explicit VerletStep(SimBox* box): SimulationStep(box),
                                    h_verletList(0),
                                    vaCoords(NULL) {}
//                                    h_verletAtomCoords(0),
//                                    d_verletList(0),
//                                    d_verletAtomCoords(0) {}

        virtual ~VerletStep();
        virtual Real calcSystemEnergy(Real &subLJ, Real &subCharge, int numMolecules);
        virtual Real calcMolecularEnergyContribution(int currMol, int startMol);
        virtual void changeMolecule(int molIdx, SimBox *box);
        virtual void rollback(int molIdx, SimBox *box);

    private:
        thrust::host_vector<int> h_verletList;
        Real* vaCoords;
//        thrust::host_vector<int> h_verletAtomCoords;
//        thrust::device_vector<int> d_verletList;
//        thrust::device_vector<int> d_verletAtomCoords;
        void createVerlet();
};

namespace VerletCalcs {
        
    /**
     *
     */
     Real calcMolecularEnergyContribution(int currMol, int startMol, thrust::host_vector<int> verletList);
    
     /**
      *
      */
    Real calcMoleculeInteractionEnergy (int m1, int m2, int* molData,
                                      Real* aData, Real* aCoords,
                                      Real* bSize, int numMols, int numAtoms);
    
    /**
     *
     */
    bool updateVerlet(Real* vaCoords, int i);
    
    /**
     *
     */
    void freeMemory(thrust::host_vector<int> &h_verletList, Real* verletAtomCoords);

    /**
     *
     */
    thrust::host_vector<int> newVerletList();
}

#endif
