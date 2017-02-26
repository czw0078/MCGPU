/*
 * A subclass of SimulationStep that uses a verlet list strategy
 */

#ifndef VERLETSTEP_H
#define VERLETSTEP_H

#include "Metropolis/SimulationStep.h"
#include "Metropolis/GPUCopy.h"

class VerletStep : public SimulationStep {

    private:
    Box* box; // Remove with transition to SimBox
    SimBox* sb;
    void copyMoleculeVerlet(Molecule* molecules, int molIndex);
    void freeVerletMolecules();
    Molecule* verletMolecules; // Remove with transition to SimBox
    int** verletList;
    int* amtOfVerletNeighbors;

    // A change to this will need a corresponding change in SimBoxBuilder.cpp
    const Real SKIN_MULTIPLIER = 1.28175;

    public:    
    VerletStep(SimBox* sb) : SimulationStep(sb), box(sb->verletListBox) {
        createVerletList(sb->numMolecules, sb->verletCutoff, false);
    }
    ~VerletStep();
    Real calcMolecularEnergyContribution(int currMol, int startMol);

    /*
     * Creates a new verlet list for each molecule in box and saves the position when the verlet list is made
     * @param isInitialized Indicates if the verlet list has been allocated previously 
     */
    void createVerletList(int numMolecules, Real verletCutoff, bool isInitialized);

    /**
     * TO BE REPLACED WITH SIMCALCS IMPLEMENTATION
     * Calculates the intermolecular energy between two molecules
     */
	Real calcInterMolecularEnergy(Molecule *molecules, int mol1, int mol2, Environment *environment);
	
    /**
     * Uses the molecule list and verletMoleculest to determine if the randomly to determine if the randomly selected 
     * molecule's displacement is greater than the skin layer of the verlet algorithm. Distance chacking is between the current 
     * position of the molecile and the position of the molecule whent he verlet list was created.
     */
    bool isOutsideSkinLayer(int moleculeIndex);
	
   /**
    * Used when calculating molecular energy contribution. We need to use the verlet list and 
    * only calculate the energy contribution of molecules within the rrcutoff range
    */
   bool isWithinRRCutoff(int molecule1, int molecule2);
	
    /**
     * TO BE REPLACED WITH SIMCALCS IMPLEMENTATION
     * Calculates the LJ energy between two atoms
     */
	Real calc_lj(Atom atom1, Atom atom2, Real r2);
	
    /**
     * TO BE REPLACED WITH SIMCALCS IMPLEMENTATION
     * Calculates the charge energy between two atoms
     */
	Real calcCharge(Real charge1, Real charge2, Real r);
	
    /**
     * TO BE REPLACED WITH SIMCALCS IMPLEMENTATION
     * Makes a distance periodic within a specified range
     */
	Real makePeriodic(Real x, Real boxDim);

    /**
     * TO BE REPLACED WITH SIMCALCS IMPLEMENTATION
     * Calculates the geometric mean of two values
     */
	Real calcBlending(Real d1, Real d2);
	
    /**
     * TO BE REPLACED WITH SIMCALCS IMPLEMENTATION
     * Calculates the squared diestance between two atoms
     */
	Real calcAtomDist(Atom atom1, Atom atom2, Environment *enviro);
};
#endif

