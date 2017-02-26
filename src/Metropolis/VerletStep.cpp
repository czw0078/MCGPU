/*
    Code adapted and modified from the "Verlet" branch in the "orlandoacevedo/MCGPU" repository 

	Note: Resources for Verlet: "Understanding Molecular Simulation: From Algorithms to Applications," by Daan Frenkel and
				     Berend Smit; page 545
*/

#include <cmath>
#include "Metropolis/VerletStep.h"
#include "Metropolis/SimulationStep.h"

Real VerletStep::calcMolecularEnergyContribution(int currMol, int startMol){
   
    // Recreate verlet list if outside skin layer
    if( VerletStep::isOutsideSkinLayer(currMol) ){
        sb = GPUCopy::simBoxCPU();
        VerletStep::createVerletList(sb->numMolecules, sb->verletCutoff, true);
    }

    Real energy = 0.0;    
    int amtOfNeighbors = amtOfVerletNeighbors[currMol];
    Molecule *molecules = box->getMolecules();
    Environment *env = box->getEnvironment();
    
    // For each neighbor the molecule has calculate the energy between them
    for(int neighbor = 0; neighbor < amtOfNeighbors; neighbor++){
        int neighborIndex = verletList[currMol][neighbor];

    	// Avoid double counting of energy calculation;moleculeCalc A to B == B to A
		if(currMol < neighborIndex) continue;

    	// Only calculate contribution with neighbors within cutoff radius NOT all verlet neighbors
    	if( isWithinRRCutoff(currMol, neighborIndex) )
            energy += calcInterMolecularEnergy(molecules, currMol, neighborIndex, env);
    }
    return energy;
}

Real VerletStep::calcInterMolecularEnergy(Molecule *molecules, int mol1, int mol2, Environment *enviro){
	Real energy = 0.0; 
    Real ljEnergy = 0.0;
    Real chargeEnergy = 0.0;

	for (int i = 0; i < molecules[mol1].numOfAtoms; i++){
		Atom atom1 = molecules[mol1].atoms[i];
		
		for (int j = 0; j < molecules[mol2].numOfAtoms; j++){
			Atom atom2 = molecules[mol2].atoms[j];
		
			if (atom1.sigma >= 0 && atom1.epsilon >= 0 && atom2.sigma >= 0 && atom2.epsilon >= 0){

				//calculate squared distance between atoms 
				Real r2 = calcAtomDist(atom1, atom2, enviro);
				
				ljEnergy = calc_lj(atom1, atom2, r2);
				chargeEnergy = calcCharge(atom1.charge, atom2.charge, sqrt(r2));

				energy += ljEnergy + chargeEnergy;
			} // if
		} // for j
	} // for i
	return energy;
}
 
void VerletStep::createVerletList(int numMolecules, Real verletCutoff, bool initialized){
	Molecule *molecules = box->getMolecules();
   	Environment *env = box->getEnvironment();
 
    // Initialize verlet list; delete old list if exists
    if(initialized) delete[] verletMolecules;
    verletMolecules = new Molecule[numMolecules];

 	for(int mol = 0; mol < numMolecules; mol++)
        VerletStep::copyMoleculeVerlet(molecules, mol);

    // Initialize the verlet list with end-of-verlet-list value = -1
    VerletStep::verletList = new int*[numMolecules];
    for(int i = 0; i < numMolecules; i++)	
        	verletList[i] = new int[numMolecules];
    
    for(int row = 0; row < numMolecules; row++)
       	for(int column = 0; column < numMolecules; column++)
       		verletList[row][column] = -1;

    // Initialize the amount of verlet neighbors
  	amtOfVerletNeighbors = new int[numMolecules];

   	// Create verlet list from approach in Book
   	bool moleculeIncluded = false;

	for(int moleculeIndexI = 0; moleculeIndexI < numMolecules; moleculeIndexI++){
       	for(int moleculeIndexJ = moleculeIndexI + 1; moleculeIndexJ < numMolecules; moleculeIndexJ++){
        	if(moleculeIndexI == moleculeIndexJ) continue;
            
       		// Grab the primary indexes of ith-Molecule and jth-Molecule
            // change to use thrust::host_vector
        	std::vector<int> ithMolPrimaryIndexArray = (*(*(env->primaryAtomIndexArray))[molecules[moleculeIndexI].type]);
           	std::vector<int> jthMolPrimaryIndexArray;
           	if (molecules[moleculeIndexI].type == molecules[moleculeIndexJ].type)
               	jthMolPrimaryIndexArray = ithMolPrimaryIndexArray;
           	else
               	jthMolPrimaryIndexArray = (*(*(env->primaryAtomIndexArray))[molecules[moleculeIndexJ].type]);
           
          	// If any of the primary indexes of a molecule are within the verlet cutoff, then add to verlet list
       		for(int primaryIndexI = 0; primaryIndexI < ithMolPrimaryIndexArray.size(); primaryIndexI++){
           		// Get an atom from molecule i
           		int atomIndex1 = ithMolPrimaryIndexArray[primaryIndexI];
           		Atom atom1 = molecules[moleculeIndexI].atoms[atomIndex1];
                
           		for(int primaryIndexJ = 0; primaryIndexJ < jthMolPrimaryIndexArray.size(); primaryIndexJ++){
           			// Get an atom from molecule j
           			int atomIndex2 = jthMolPrimaryIndexArray[primaryIndexJ];
           			Atom atom2 = molecules[moleculeIndexJ].atoms[atomIndex2];
                   
           			// Get the distancebetween atom1 and atom2
          			if( VerletStep::calcAtomDist(atom1, atom2, env)  < verletCutoff ){
              			// add moleculeJ as moleculeI's neighbor and vice versa
              			int neighborI = amtOfVerletNeighbors[moleculeIndexI];
               			verletList[moleculeIndexI][neighborI] = moleculeIndexJ;
               			int neighborJ = amtOfVerletNeighbors[moleculeIndexJ];
              			verletList[moleculeIndexJ][neighborJ] = moleculeIndexI;
                     
               			// Increase the said amount of neighbors for ith & jth molecule
                        amtOfVerletNeighbors[moleculeIndexI]++;
                    	amtOfVerletNeighbors[moleculeIndexJ]++;
            			moleculeIncluded = true;
                        break;
            		} // if
        		} // primaryIndexJ

                // if true, exit because we're done looking at primary atoms of molecule i
           		if(moleculeIncluded) break;

       	    } // primaryIndexI
      	} // moleculeIndexJ
 	} // moleculeIndexI   
} // createVerletList()
 
bool VerletStep::isOutsideSkinLayer(int moleculeIndex){
    Environment *env = box->getEnvironment();
    
    // Get the same molecule, but at possibly different positions in the box due to past movements
    Molecule *moleculePosition1 = box->getMolecules();
    Molecule *moleculePosition2 = verletMolecules;
    
    // Get the primary indexes. They should be the same
    std::vector<int> molecule1PrimaryIndexes = (*(*(env->primaryAtomIndexArray))[moleculePosition1[moleculeIndex].type]);
    std::vector<int> molecule2PrimaryIndexes = (*(*(env->primaryAtomIndexArray))[moleculePosition2[moleculeIndex].type]);
    
    // For all the primary index atoms calculate the distance between the two molecules.

    // See resource note at top for information on cutoffs
    //   verletSkinLayer = ( verletCutoff                         - radiusCutoff        ) / 2.0 
    Real verletSkinLayer = ( GPUCopy::simBoxCPU()->verletCutoff   - (GPUCopy::simBoxCPU()->verletCutoff / SKIN_MULTIPLIER) ) / 2.0;

    for(int i = 0; i < molecule1PrimaryIndexes.size(); i++){
        // Get an atom from molecule1
        int primaryIndex1 = molecule1PrimaryIndexes[i];
        Atom atom1 = moleculePosition1[moleculeIndex].atoms[primaryIndex1];
        
        for(int j = 0; j < molecule2PrimaryIndexes.size(); j++){
            // Get an atom from molecule2
            int primaryIndex2 = molecule2PrimaryIndexes[j];
            Atom atom2 = moleculePosition2[moleculeIndex].atoms[primaryIndex2];

            if( VerletStep::calcAtomDist(atom1, atom2, env) > verletSkinLayer)
                return true;
        } // for j
    } // for i
    return false;
} // isOutsideSkinLayer()

bool VerletStep::isWithinRRCutoff(int molecule1, int molecule2){
	Molecule *molecules = box->getMolecules();
	Environment *env = box->getEnvironment();
	Real rrCut = GPUCopy::simBoxCPU()->verletCutoff / SKIN_MULTIPLIER;

	// Get the primary atom index array
	std::vector<int> moleculeAPrimaries = (*(*(env->primaryAtomIndexArray))[molecules[molecule1].type]);
	std::vector<int> moleculeBPrimaries;
	if(molecules[molecule1].type == molecules[molecule2].type)
		moleculeBPrimaries = moleculeAPrimaries;
	else
		moleculeBPrimaries = (*(*(env->primaryAtomIndexArray))[molecules[molecule2].type]);
        
	// for each molecule's primary atoms check the distance between the two molecules
	for(int primaryAtomA = 0; primaryAtomA < moleculeAPrimaries.size(); primaryAtomA++){
		int atomIndexA = moleculeAPrimaries[primaryAtomA];
		Atom atomA = molecules[molecule1].atoms[atomIndexA];
		
		for(int primaryAtomB = 0; primaryAtomB < moleculeBPrimaries.size(); primaryAtomB++){
			int atomIndexB = moleculeBPrimaries[primaryAtomB];
			Atom atomB = molecules[molecule2].atoms[atomIndexB];

			if( VerletStep::calcAtomDist(atomA, atomB, env) < rrCut)
				return true;
		} // for primaryAtomB
	} // for primaryAtomA
	return false;
}

void VerletStep::copyMoleculeVerlet(Molecule* molecules, int molIndex){
    Molecule* sourceMolecule = &molecules[molIndex];

    // Free memory of molecule before allocating
    if( verletMolecules[molIndex].numOfAtoms != 0 ){
        delete[] verletMolecules[molIndex].atoms;
        delete[] verletMolecules[molIndex].bonds;
        delete[] verletMolecules[molIndex].angles;
        delete[] verletMolecules[molIndex].dihedrals;
        delete[] verletMolecules[molIndex].hops;
    }

    memcpy( &verletMolecules[molIndex], sourceMolecule, box->getChangedMolSize() );

    verletMolecules[molIndex].atoms = new Atom[sourceMolecule->numOfAtoms];
    verletMolecules[molIndex].bonds = new Bond[sourceMolecule->numOfBonds];
    verletMolecules[molIndex].angles = new Angle[sourceMolecule->numOfAngles];
    verletMolecules[molIndex].dihedrals = new Dihedral[sourceMolecule->numOfDihedrals];
    verletMolecules[molIndex].hops = new Hop[sourceMolecule->numOfHops];

    box->copyMolecule(&verletMolecules[molIndex], sourceMolecule);
}

Real VerletStep::calc_lj(Atom atom1, Atom atom2, Real r2){
    if (r2 == 0.0) return 0.0;
    
    Real sigma = calcBlending(atom1.sigma, atom2.sigma);
    Real epsilon = calcBlending(atom1.epsilon, atom2.epsilon);
    
    Real sig2OverR2 = pow(sigma, 2) / r2;
	Real sig6OverR6 = pow(sig2OverR2, 3);
    Real sig12OverR12 = pow(sig6OverR6, 2);
    return 4.0 * epsilon * (sig12OverR12 - sig6OverR6);
}

Real VerletStep::calcCharge(Real charge1, Real charge2, Real r){ 
    if (r == 0.0) return 0.0;
    // 332.06 conversion factor for units in kcal/mol
    return (charge1 * charge2 * 332.06) / r;
}

Real VerletStep::makePeriodic(Real x, Real boxDim){
    if(x < -0.5 * boxDim) x += boxDim;
    else if(x > 0.5 * boxDim) x -= boxDim;
    return x;
} 

Real VerletStep::calcBlending(Real d1, Real d2){
    return sqrt(d1 * d2);
}

Real VerletStep::calcAtomDist(Atom atom1, Atom atom2, Environment *enviro){
	// Calculate difference in coordinates
	Real deltaX = makePeriodic(atom1.x - atom2.x, enviro->x);
	Real deltaY = makePeriodic(atom1.y - atom2.y, enviro->y);
	Real deltaZ = makePeriodic(atom1.z - atom2.z, enviro->z);
				
	// Calculate squared distance (r2 value) and return
	return pow(deltaX, 2) + pow(deltaY, 2) + pow(deltaZ, 2);
}

VerletStep::~VerletStep(){
    delete[] verletMolecules;
    delete[] verletList;
    delete[] amtOfVerletNeighbors;
}








