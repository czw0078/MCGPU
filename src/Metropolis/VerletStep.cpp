#include "VerletStep.h"
#include "ProximityMatrixStep.h"
#include "SimulationStep.h"
#include "GPUCopy.h"

Real VerletStep::calcMolecularEnergyContribution(int currMol, int startMol) {
    return VerletCalcs::calcMolecularEnergyContribution(currMol, startMol, this->h_verletList);
}

Real VerletStep::calcSystemEnergy(Real &subLJ, Real &subCharge, int numMolecules) {
    Real result = SimulationStep::calcSystemEnergy(subLJ, subCharge, numMolecules);
    VerletStep::createVerlet();
    return result;
}

void VerletStep::createVerlet(){

    SimBox* sb = GPUCopy::simBoxCPU();
    Real *atomCoords = sb->atomCoordinates;
    const int numAtoms = sb->numAtoms;

    VerletCalcs::freeMemory(this->h_verletList, this->vaCoords);

    this->h_verletList = VerletCalcs::newVerletList();

    vaCoords = new Real[NUM_DIMENSIONS * numAtoms];
    for(int i = 0; i < NUM_DIMENSIONS * numAtoms; i++)
        vaCoords[i] = atomCoords[i];
} // createVerlet

void VerletStep::changeMolecule(int molIdx, SimBox *box) {
    SimulationStep::changeMolecule(molIdx, box);
    if( VerletCalcs::updateVerlet(this->vaCoords, molIdx) )
        VerletStep::createVerlet();
}

void VerletStep::rollback(int molIdx, SimBox *box) {
    SimulationStep::rollback(molIdx, box);
    if( VerletCalcs::updateVerlet(this->vaCoords, molIdx) )
        VerletStep::createVerlet();
}

VerletStep::~VerletStep() {
  VerletCalcs::freeMemory(this->h_verletList, this->vaCoords);
  this->vaCoords = NULL;
}


// ----- VerletCalcs Definitions -----

Real VerletCalcs::calcMolecularEnergyContribution(int currMol, int startMol, thrust::host_vector<int> verletList){
    Real total = 0.0;

    SimBox* sb = GPUCopy::simBoxCPU();
    int *molData = sb->moleculeData;
    Real *atomCoords = sb->atomCoordinates;
    Real *bSize = sb->size;
    int *pIdxes = sb->primaryIndexes;
    Real *aData = sb->atomData;
    Real cutoff = sb->cutoff;
    const long numMolecules = sb->numMolecules;
    const int numAtoms = sb->numAtoms;

    const int p1Start = molData[MOL_PIDX_START * numMolecules + currMol];
    const int p1End = molData[MOL_PIDX_COUNT * numMolecules + currMol] + p1Start;

    if (verletList.empty()) {
        for (int otherMol = startMol; otherMol < numMolecules; otherMol++) {
            if (otherMol != currMol) {
                int p2Start = molData[MOL_PIDX_START * numMolecules + otherMol];
                int p2End = molData[MOL_PIDX_COUNT * numMolecules + otherMol] + p2Start;
                if (SimCalcs::moleculesInRange(p1Start, p1End, p2Start, p2End,
                                       atomCoords, bSize, pIdxes, cutoff, numAtoms)) {
                    total += calcMoleculeInteractionEnergy(currMol, otherMol, molData,
                                                 aData, atomCoords, bSize,
                                                 numMolecules, numAtoms);
                }
            }
        }
    } else {
        int neighborIndex = verletList[currMol * numMolecules];
        for(int i = 0; neighborIndex != -1; i++){
            neighborIndex = verletList[currMol * numMolecules + i];

            if( neighborIndex == currMol  || neighborIndex == -1 ) continue;

            int p2Start = molData[MOL_PIDX_START * numMolecules + neighborIndex];
            int p2End = molData[MOL_PIDX_COUNT * numMolecules + neighborIndex] + p2Start;

            if (SimCalcs::moleculesInRange(p1Start, p1End, p2Start, p2End,
                                atomCoords, bSize, pIdxes, cutoff, numAtoms)) {

                total += calcMoleculeInteractionEnergy(currMol, neighborIndex, molData,
                               aData, atomCoords, bSize, numMolecules, numAtoms);
            } // if
        } // for neighbors 
    } // else
    return total;
} // calcMolecularEnergyContribution()

Real VerletCalcs::calcMoleculeInteractionEnergy(int m1, int m2, int* molData, Real* aData, Real* aCoords,
                                                    Real* bSize, int numMolecules, int numAtoms) {
    Real energySum = 0;

    const int m1Start = molData[MOL_START * numMolecules + m1];
    const int m1End = molData[MOL_LEN * numMolecules + m1] + m1Start;

    const int m2Start = molData[MOL_START * numMolecules + m2];
    const int m2End = molData[MOL_LEN * numMolecules + m2] + m2Start;

    for (int i = m1Start; i < m1End; i++) {
        for (int j = m2Start; j < m2End; j++) {
            if (aData[ATOM_SIGMA * numAtoms + i] >= 0 && aData[ATOM_SIGMA * numAtoms + j] >= 0
                && aData[ATOM_EPSILON * numAtoms + i] >= 0 && aData[ATOM_EPSILON * numAtoms + j] >= 0) {

                const Real r2 = SimCalcs::calcAtomDistSquared(i, j, aCoords, bSize, numAtoms);
                if (r2 == 0.0) {
                    energySum += 0.0;
                } else {
                    energySum += SimCalcs::calcLJEnergy(i, j, r2, aData, numAtoms);
                    energySum += SimCalcs::calcChargeEnergy(i, j, sqrt(r2), aData, numAtoms);
                }
            }
        }
    }
  return energySum;
}

bool VerletCalcs::updateVerlet(Real* vaCoords, int i) {
    SimBox* sb = GPUCopy::simBoxCPU();
    const Real cutoff = pow(sb->cutoff, 2);

    Real* atomCoords = sb->atomCoordinates;
    Real* bSize = sb->size;
    int numAtoms = sb->numAtoms;

    Real dx = SimCalcs::makePeriodic(atomCoords[X_COORD * numAtoms + i] -  vaCoords[X_COORD * numAtoms + i], X_COORD, bSize);
    Real dy = SimCalcs::makePeriodic(atomCoords[Y_COORD * numAtoms + i] -  vaCoords[Y_COORD * numAtoms + i], Y_COORD, bSize);
    Real dz = SimCalcs::makePeriodic(atomCoords[Z_COORD * numAtoms + i] -  vaCoords[Z_COORD * numAtoms + i], Z_COORD, bSize);

    Real dist = pow(dx, 2) + pow(dy, 2) + pow(dz, 2);
    if( cutoff < dist )
        return true;
    return false;
} // updateVerlet()

thrust::host_vector<int> VerletCalcs::newVerletList(){
    SimBox* sb = GPUCopy::simBoxCPU();
    int *molData = sb->moleculeData;
    Real *atomCoords = sb->atomCoordinates;
    Real *bSize = sb->size;
    int *pIdxes = sb->primaryIndexes;
    Real cutoff = sb->cutoff * sb->cutoff;
    const long numMolecules = sb->numMolecules;
    const int numAtoms = sb->numAtoms;
    int p1Start, p1End;
    int p2Start, p2End;

    thrust::host_vector<int> verletList(numMolecules * numMolecules, -1);

    int numNeighbors;
    for(int i = 0; i < numMolecules; i++){

        numNeighbors = 0;
        p1Start = molData[MOL_PIDX_START * numMolecules + i];
        p1End = molData[MOL_PIDX_COUNT * numMolecules + i] + p1Start;

        for(int j = 0; j < numMolecules; j++){
            verletList[i * numMolecules + j] = -1;

            if (i != j) {
                p2Start = molData[MOL_PIDX_START * numMolecules + j];
                p2End = molData[MOL_PIDX_COUNT * numMolecules + j] + p2Start;

                if (SimCalcs::moleculesInRange(p1Start, p1End, p2Start, p2End,
                                       atomCoords, bSize, pIdxes, cutoff, numAtoms)) {

                    verletList[i * numMolecules + numNeighbors ] = j;
                    numNeighbors++;
                } // if in range
            }
        } // for molecule j
    } // for molecule i
    return verletList;
} // newVerletList()

void VerletCalcs::freeMemory(thrust::host_vector<int> &verletList, Real* verletAtomCoords) {
    verletList.clear();
    verletList.shrink_to_fit();
    if( verletAtomCoords != NULL )
        delete[] verletAtomCoords;
}

