#include "VerletStep.h"
#include "ProximityMatrixStep.h"
#include "SimulationStep.h"
#include "GPUCopy.h"

#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#define BLOCKS 4
#define THREADS 32

Real VerletStep::calcMolecularEnergyContribution(int currMol, int startMol) {
    Real energy = 0.0;
    thrust::plus<Real> sum;
    Real init = 0.0;
    int begIdx, endIdx;

    if(GPUCopy::onGpu()) {
        VerletCalcs::EnergyContribution<int> Contribution( currMol, startMol, GPUCopy::simBoxGPU() );
        // Verlet list size check to handle initial energy calculation
        if( this->d_verletList.size() != this->NUM_MOLS ) {
            begIdx = currMol * this->NUM_MOLS;
            endIdx = begIdx + this->NUM_MOLS;
            energy = thrust::transform_reduce( &this->d_verletList[begIdx],
                                               &this->d_verletList[endIdx],
                                               Contribution, init, sum );
        } else {
            energy = thrust::transform_reduce( this->d_verletList.begin(),
                                               this->d_verletList.end(),
                                               Contribution, init, sum );
            energy /= 2;
        } // Initial system energy
        cudaDeviceSynchronize();
        return energy;
    } else {    // on CPU
        VerletCalcs::EnergyContribution<int> Contribution( currMol, startMol, GPUCopy::simBoxCPU() );
        // Verlet list size check to handle initial energy calculation
        if( this->h_verletList.size() != this->NUM_MOLS ) {
            begIdx = currMol * this->NUM_MOLS;
            endIdx = begIdx + this->NUM_MOLS;
            energy = thrust::transform_reduce( &this->h_verletList[begIdx],
                                               &this->h_verletList[endIdx],
                                               Contribution, init, sum );
        } else {
            energy = thrust::transform_reduce( this->h_verletList.begin(),
                                               this->h_verletList.end(),
                                               Contribution, init, sum );
            energy /= 2;
        } // Initial system energy
    } // else CPU
    return energy;
} // calcMolecularEnergyContribution

void VerletStep::checkOutsideSkinLayer(int molIdx) {
    VerletCalcs::UpdateVerletList<int> update;
    bool refresh;

    if( GPUCopy::onGpu() ) {
//        std::cout << "0" << std::endl;
        //update( molIdx, (Real *) thrust::raw_pointer_cast( &this->d_verletAtomCoords[0] ), GPUCopy::simBoxGPU() );
        VerletCalcs::updateKernel<<<BLOCKS,THREADS>>>( molIdx, (Real *) thrust::raw_pointer_cast( &this->d_verletAtomCoords[0] ), GPUCopy::simBoxGPU() );
//        std::cout << "1" << std::endl;
        cudaDeviceSynchronize();
//        std::cout << "2" << std::endl;
        cudaMemcpy( &refresh, &(GPUCopy::simBoxGPU()->updateVerlet), sizeof(bool), cudaMemcpyDeviceToHost );
//        std::cout << "3" << std::endl;

        if( refresh )
            VerletStep::CreateVerletList();
    } else {
        update( molIdx, &this->h_verletAtomCoords[0], GPUCopy::simBoxCPU() );
        if( GPUCopy::simBoxCPU()->updateVerlet )
            VerletStep::CreateVerletList();
    }
} // // checkOutsideSkinLayer

void VerletStep::CreateVerletList() {
    VerletStep::freeMemory();
    VerletStep::resizeThrustVectors();

    if( GPUCopy::onGpu() ) {
        VerletCalcs::createVerlet<<<1,1>>>( (int *) thrust::raw_pointer_cast(&this->d_verletList[0]), (Real* ) thrust::raw_pointer_cast(&this->d_verletAtomCoords[0]),
                                    this->d_verletList.size(), this->d_verletAtomCoords.size(),
                                    GPUCopy::simBoxGPU() );
    } else {
        // Create new verlet list
        int* vList = VerletCalcs::newVerletList( GPUCopy::simBoxCPU() , this->VERLET_SIZE );
        thrust::copy( thrust::host, &vList[0], &vList[0] + this->VERLET_SIZE, this->h_verletList.begin() );

        // Copy new base atom coordinates
        thrust::copy( thrust::host,
                      &GPUCopy::simBoxCPU()->atomCoordinates[0],
                      &GPUCopy::simBoxCPU()->atomCoordinates[0] + this->VACOORDS_SIZE,
                      this->h_verletAtomCoords.begin() );

    } // else CPU
} // CreateVerletList

Real VerletStep::calcSystemEnergy(Real &subLJ, Real &subCharge, int numMolecules) {

    if(GPUCopy::onGpu()) {
        this->d_verletList.resize( this->NUM_MOLS );
        thrust::sequence( thrust::device, this->d_verletList.begin(), this->d_verletList.end() );
    } else {
        this->h_verletList.resize( this->NUM_MOLS );
        thrust::sequence( thrust::host, this->h_verletList.begin(), this->h_verletList.end() );
    }
    Real result = SimulationStep::calcSystemEnergy(subLJ, subCharge, numMolecules);
    VerletStep::CreateVerletList();
    return result;
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

void VerletStep::changeMolecule(int molIdx, SimBox *box) {
    SimulationStep::changeMolecule(molIdx, box);
    VerletStep::checkOutsideSkinLayer(molIdx);
} // changeMolecule

void VerletStep::rollback(int molIdx, SimBox *box) {
    SimulationStep::rollback(molIdx, box);
    VerletStep::checkOutsideSkinLayer(molIdx);
} // rollback

VerletStep::~VerletStep() {
    VerletStep::freeMemory();
}

// ----- VerletCalcs Definitions -----
template <typename T>
Real VerletCalcs::EnergyContribution<T>::operator()( const T neighborIndex ) const {
    Real total = 0.0;

    // Exit if neighborIndex is currMol or is not a verlet neighbor
    if( neighborIndex == -1 || neighborIndex == currMol )
        return total;

    int *molData = sb->moleculeData;
    Real *atomCoords = sb->atomCoordinates;
    Real *bSize = sb->size;
    int *pIdxes = sb->primaryIndexes;
    Real cutoff = sb->cutoff;
    const long numMolecules = sb->numMolecules;
    const int numAtoms = sb->numAtoms;

    const int p1Start = molData[MOL_PIDX_START * numMolecules + currMol];
    const int p1End = molData[MOL_PIDX_COUNT * numMolecules + currMol] + p1Start;

    const int p2Start = molData[MOL_PIDX_START * numMolecules + neighborIndex];
    const int p2End = molData[MOL_PIDX_COUNT * numMolecules + neighborIndex] + p2Start;

    if (SimCalcs::moleculesInRange(p1Start, p1End, p2Start, p2End,
                                   atomCoords, bSize, pIdxes, cutoff, numAtoms))
      total += calcMoleculeInteractionEnergy(currMol, neighborIndex, sb);
    return total;
} // EnergyContribution

template <typename T>
void VerletCalcs::UpdateVerletList<T>::operator()( const T i, const Real* vaCoords, SimBox* sb ) const {
    const Real cutoff = sb->cutoff * sb->cutoff;
    sb->updateVerlet = false;

    Real* atomCoords = sb->atomCoordinates;
    Real* bSize = sb->size;
    int numAtoms = sb->numAtoms;

    Real dx = SimCalcs::makePeriodic(atomCoords[X_COORD * numAtoms + i] -  vaCoords[X_COORD * numAtoms + i], X_COORD, bSize);
    Real dy = SimCalcs::makePeriodic(atomCoords[Y_COORD * numAtoms + i] -  vaCoords[Y_COORD * numAtoms + i], Y_COORD, bSize);
    Real dz = SimCalcs::makePeriodic(atomCoords[Z_COORD * numAtoms + i] -  vaCoords[Z_COORD * numAtoms + i], Z_COORD, bSize);

    Real dist = pow(dx, 2) + pow(dy, 2) + pow(dz, 2);
    sb->updateVerlet = dist > cutoff;
} // UpdateVerletList


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
} // calcMoleculeInteractionEnergy

__global__
void VerletCalcs::createVerlet(int* verletList, Real* verletAtomCoords, int verletListLength, int vaCoordsLength, SimBox* sb){

    // Initialize a new verlet list
    int* vList = VerletCalcs::newVerletList(sb, verletListLength);

    for(int i = 0; i < verletListLength; i++)
        verletList[i] = vList[i];

    // Copy data for base atom coordinates
    for(int i = 0; i < vaCoordsLength; i++)
        verletAtomCoords[i] = sb->atomCoordinates[i];
} // createVerlet

template <typename T>
T* VerletCalcs::NewVerletList<T>::operator()() {
    int *molData = sb->moleculeData;
    Real *atomCoords = sb->atomCoordinates;
    Real *bSize = sb->size;
    int *pIdxes = sb->primaryIndexes;
    Real cutoff = sb->cutoff * sb->cutoff;
    const long numMolecules = sb->numMolecules;
    const int numAtoms = sb->numAtoms;
    int p1Start, p1End;
    int p2Start, p2End;

    T* verletList = new T[sb->numMolecules * sb->numMolecules];

    int numNeighbors;
    for(T i = 0; i < numMolecules; i++){

        numNeighbors = 0;
        p1Start = molData[MOL_PIDX_START * numMolecules + i];
        p1End = molData[MOL_PIDX_COUNT * numMolecules + i] + p1Start;

        for(T j = 0; j < numMolecules; j++){
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
} // NewVerletList

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

void VerletStep::freeMemory() {
    h_verletList.clear();
    h_verletList.shrink_to_fit();
    h_verletAtomCoords.clear();
    h_verletAtomCoords.shrink_to_fit();

    d_verletList.clear();
    d_verletList.shrink_to_fit();
    d_verletAtomCoords.clear();
    d_verletAtomCoords.shrink_to_fit();
}

