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

class VerletStep: public SimulationStep {
    public:
        explicit VerletStep(SimBox* box): SimulationStep(box),
                                    vl(NULL),
                                    vaCoords(NULL) {}


        virtual ~VerletStep();
        virtual Real calcSystemEnergy(Real &subLJ, Real &subCharge, int numMolecules);
        virtual Real calcMolecularEnergyContribution(int currMol, int startMol);
        virtual void changeMolecule(int molIdx, SimBox *box);
        virtual void rollback(int molIdx, SimBox *box);

    private:
        int* vl;
        Real* vaCoords;
        void createVerlet();
};

namespace VerletCalcs {
        
    /**
     *
     */
     Real calcMolecularEnergyContribution(int currMol, int startMol, int* verletList);
    
     /**
      *
      */
    Real calcMoleculeInteractionEnergy (int m1, int m2, int* molData,
                                      Real* aData, Real* aCoords,
                                      Real* bSize, int numMols, int numAtoms);
    
    /**
     *
     */
    bool updateVerlet(Real* verletList, int i);
    
    /**
     *
     */
    void freeMemory(int* verletList, Real* verletAtomCoords);

    /**
     *
     */
    int* newVerletList();
}

#endif
