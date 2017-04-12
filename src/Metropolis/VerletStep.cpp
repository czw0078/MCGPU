#include "VerletStep.h"
#include "ProximityMatrixStep.h"
#include "SimulationStep.h"
#include "GPUCopy.h"

Real VerletStep::calcMolecularEnergyContribution(int currMol, int startMol) {
    if(GPUCopy::onGpu()) {
        Real energy = 0.0;
        cudaDeviceSynchronize();

        return energy;
    } else {    // on CPU
        VerletCalcs::calcMolecularEnergyContribution(currMol, startMol, GPUCopy::simBoxCPU(), (int *) &this->h_verletList[0], this->h_verletList.size());
        return GPUCopy::simBoxCPU()->energy;
    }
}

void VerletStep::resizeThrustVectors() {
    if(GPUCopy::onGpu()) {
        this->d_verletList.resize(this->VERLET_SIZE);
        this->d_verletAtomCoords.resize(this->VACOORDS_SIZE);
    } else {    // on CPU
        this->h_verletList.resize(this->VERLET_SIZE);
        this->h_verletAtomCoords.resize(this->VACOORDS_SIZE);
    }
}

Real VerletStep::calcSystemEnergy(Real &subLJ, Real &subCharge, int numMolecules) {
    Real result = SimulationStep::calcSystemEnergy(subLJ, subCharge, numMolecules);
    VerletStep::resizeThrustVectors();

    if(GPUCopy::onGpu()) {
        VerletCalcs::createVerlet((int *) thrust::raw_pointer_cast(&this->d_verletList[0]), (Real* ) thrust::raw_pointer_cast(&this->d_verletAtomCoords[0]),
                                    this->d_verletList.size(), this->d_verletAtomCoords.size(),
                                    GPUCopy::simBoxGPU() );
    } else {    // on CPU
        VerletCalcs::createVerlet((int *) &this->h_verletList[0], (Real* ) &this->h_verletAtomCoords[0],
                                    this->h_verletList.size(), this->h_verletAtomCoords.size(),
                                    GPUCopy::simBoxCPU() );
    }
    return result;
}


void VerletStep::changeMolecule(int molIdx, SimBox *box) {
    SimulationStep::changeMolecule(molIdx, box);

    if(GPUCopy::onGpu()) {
        // GPU implementation

    } else {    // on CPU
        VerletCalcs::updateVerlet((Real *) &this->h_verletAtomCoords[0], GPUCopy::simBoxCPU(), molIdx);
        if( GPUCopy::simBoxCPU()->updateVerlet ) {
            VerletCalcs::freeMemory(this->h_verletList, this->h_verletAtomCoords);
            VerletStep::resizeThrustVectors();
            VerletCalcs::createVerlet((int *) &this->h_verletList[0], (Real* ) &this->h_verletAtomCoords[0],
                                this->h_verletList.size(), this->h_verletAtomCoords.size(),
                                GPUCopy::simBoxCPU() );

        }
    }
}

void VerletStep::rollback(int molIdx, SimBox *box) {
    SimulationStep::rollback(molIdx, box);

    if(GPUCopy::onGpu()) {
        // GPU implementation

    } else {    // on CPU
        VerletCalcs::updateVerlet((Real *) &this->h_verletAtomCoords[0], GPUCopy::simBoxCPU(), molIdx);
        if( GPUCopy::simBoxCPU()->updateVerlet ) {
            VerletCalcs::freeMemory(this->h_verletList, this->h_verletAtomCoords);
            VerletStep::resizeThrustVectors();
            VerletCalcs::createVerlet((int *) &this->h_verletList[0], (Real* ) &this->h_verletAtomCoords[0],
                                this->h_verletList.size(), this->h_verletAtomCoords.size(),
                                GPUCopy::simBoxCPU() );
        }
    }
}

VerletStep::~VerletStep() {
    VerletCalcs::freeMemory(this->h_verletList, this->h_verletAtomCoords);
    VerletCalcs::freeMemory(this->d_verletList, this->d_verletAtomCoords);
}


// ----- VerletCalcs Definitions -----

void VerletCalcs::calcMolecularEnergyContribution(int currMol, int startMol, SimBox* sb, int* verletList, int verletListLength){
    Real total = 0.0;

    int *molData = sb->moleculeData;
    Real *atomCoords = sb->atomCoordinates;
    Real *bSize = sb->size;
    int *pIdxes = sb->primaryIndexes;
    Real cutoff = sb->cutoff;
    const long numMolecules = sb->numMolecules;
    const int numAtoms = sb->numAtoms;

    const int p1Start = molData[MOL_PIDX_START * numMolecules + currMol];
    const int p1End = molData[MOL_PIDX_COUNT * numMolecules + currMol] + p1Start;

    if (verletListLength == 0) {
        for (int otherMol = startMol; otherMol < numMolecules; otherMol++) {
            if (otherMol != currMol) {
                int p2Start = molData[MOL_PIDX_START * numMolecules + otherMol];
                int p2End = molData[MOL_PIDX_COUNT * numMolecules + otherMol] + p2Start;
                if (SimCalcs::moleculesInRange(p1Start, p1End, p2Start, p2End,
                                       atomCoords, bSize, pIdxes, cutoff, numAtoms)) {
                    total += calcMoleculeInteractionEnergy(currMol, otherMol, sb);
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

                total += calcMoleculeInteractionEnergy(currMol, neighborIndex, sb);
            } // if
        } // for neighbors
    } // else
    sb->energy = total;
} // calcMolecularEnergyContribution()

__global__
void VerletCalcs::energyContribution_Kernel(int currMol, int startMol, SimBox* sb, int* verletList, int verletListLength) {
    VerletCalcs::calcMolecularEnergyContribution(currMol, startMol, sb, verletList, verletListLength);
} // energyContribution_Kernel

__host__ __device__
Real VerletCalcs::calcMoleculeInteractionEnergy(int m1, int m2, SimBox* sb) {
    Real energySum = 0;

    int *molData = sb->moleculeData;
    Real *atomCoords = sb->atomCoordinates;
    Real *bSize = sb->size;
    Real *aData = sb->atomData;
    const long numMolecules = sb->numMolecules;
    const int numAtoms = sb->numAtoms;

    const int m1Start = molData[MOL_START * numMolecules + m1];
    const int m1End = molData[MOL_LEN * numMolecules + m1] + m1Start;

    const int m2Start = molData[MOL_START * numMolecules + m2];
    const int m2End = molData[MOL_LEN * numMolecules + m2] + m2Start;

    for (int i = m1Start; i < m1End; i++) {
        for (int j = m2Start; j < m2End; j++) {
            if (aData[ATOM_SIGMA * numAtoms + i] >= 0 && aData[ATOM_SIGMA * numAtoms + j] >= 0
                && aData[ATOM_EPSILON * numAtoms + i] >= 0 && aData[ATOM_EPSILON * numAtoms + j] >= 0) {

                const Real r2 = SimCalcs::calcAtomDistSquared(i, j, atomCoords, bSize, numAtoms);
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

__host__ __device__
void VerletCalcs::createVerlet(int* verletList, Real* verletAtomCoords, int verletListLength, int vaCoordsLength, SimBox* sb){

    // Initialize a new verlet list
    int* vList = VerletCalcs::newVerletList(sb, verletListLength);

    for(int i = 0; i < verletListLength; i++)
        verletList[i] = vList[i];

    // Copy data for base atom coordinates
    for(int i = 0; i < vaCoordsLength; i++)
        verletAtomCoords[i] = sb->atomCoordinates[i];
} // createVerlet

void VerletCalcs::updateVerlet(Real* vaCoords, SimBox* sb, int i) {
    const Real cutoff = pow(sb->cutoff, 2);
    sb->updateVerlet = false;

    Real* atomCoords = sb->atomCoordinates;
    Real* bSize = sb->size;
    int numAtoms = sb->numAtoms;

    Real dx = SimCalcs::makePeriodic(atomCoords[X_COORD * numAtoms + i] -  vaCoords[X_COORD * numAtoms + i], X_COORD, bSize);
    Real dy = SimCalcs::makePeriodic(atomCoords[Y_COORD * numAtoms + i] -  vaCoords[Y_COORD * numAtoms + i], Y_COORD, bSize);
    Real dz = SimCalcs::makePeriodic(atomCoords[Z_COORD * numAtoms + i] -  vaCoords[Z_COORD * numAtoms + i], Z_COORD, bSize);

    Real dist = pow(dx, 2) + pow(dy, 2) + pow(dz, 2);
    if( cutoff < dist )
        sb->updateVerlet = true;;
} // updateVerlet()

__host__ __device__
int* VerletCalcs::newVerletList(SimBox* sb, int verletListLength){
    int *molData = sb->moleculeData;
    Real *atomCoords = sb->atomCoordinates;
    Real *bSize = sb->size;
    int *pIdxes = sb->primaryIndexes;
    Real cutoff = sb->cutoff * sb->cutoff;
    const long numMolecules = sb->numMolecules;
    const int numAtoms = sb->numAtoms;
    int p1Start, p1End;
    int p2Start, p2End;

    int* verletList = new int[verletListLength];

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

__global__
void newVerletList_Kernel(int* &verletList, int verletListLength) {
 
}

void VerletCalcs::freeMemory(thrust::host_vector<int> &h_verletList, thrust::host_vector<Real> &h_verletAtomCoords) {
    h_verletList.clear();
    h_verletList.shrink_to_fit();
    h_verletAtomCoords.clear();
    h_verletAtomCoords.shrink_to_fit();
}

void VerletCalcs::freeMemory(thrust::device_vector<int> &d_verletList, thrust::device_vector<Real> &d_verletAtomCoords) {
    d_verletList.clear();
    d_verletList.shrink_to_fit();
    d_verletAtomCoords.clear();
    d_verletAtomCoords.shrink_to_fit();
}

