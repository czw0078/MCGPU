#include "GPUCopy.h"

bool launchOnGpu = false;

SimBox* h_sb = NULL;
SimBox* d_sb = NULL;

Real* h_atomData = NULL;
Real* d_atomData = NULL;

Real* h_rollBackCoordinates = NULL;
Real* d_rollBackCoordinates = NULL;

Real** h_atomCoordinates = NULL;
Real** d_atomCoordinates = NULL;

int* h_primaryIndexes = NULL;
int* d_primaryIndexes = NULL;

int* h_moleculeData = NULL;
int* d_moleculeData = NULL;

Real* h_size = NULL;
Real* d_size = NULL;

// Angle values

Real** h_angleData = NULL;
Real** d_angleData = NULL;

Real* h_angleSizes = NULL;
Real* d_angleSizes = NULL;

Real* h_rollBackAngleSizes = NULL;
Real* d_rollBackAngleSizes = NULL;

// Bond values

Real** h_bondData = NULL;
Real** d_bondData = NULL;

Real* h_bondLengths = NULL;
Real* d_bondLengths = NULL;

Real* h_rollBackBondLengths = NULL;
Real* d_rollBackBondLengths = NULL;

bool GPUCopy::onGpu() {
  return launchOnGpu;
}

void GPUCopy::setParallel(bool launchOnGpu_in) {
  launchOnGpu = launchOnGpu_in;
}

SimBox* GPUCopy::simBoxGPU() {
  return d_sb;
}

SimBox* GPUCopy::simBoxCPU() {
  return h_sb;
}

void GPUCopy::copyIn(SimBox *sb) {
  h_sb = sb;

  cudaMalloc(&d_sb, sizeof(SimBox));
  assert(d_sb != NULL);
  cudaMemcpy(d_sb, h_sb, sizeof(SimBox), cudaMemcpyHostToDevice);

  // Copy in moleculeData
  cudaMalloc(&d_moleculeData, MOL_DATA_SIZE * sizeof(int) * sb->numMolecules);
  assert(d_moleculeData != NULL);
  cudaMemcpy(d_moleculeData, h_moleculeData, MOL_DATA_SIZE * sizeof(int) *
    sb->numMolecules, cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sb->moleculeData), &d_moleculeData, sizeof(int*),
    cudaMemcpyHostToDevice);

  // Copy in atomData
  cudaMalloc(&d_atomData, ATOM_DATA_SIZE * sizeof(Real) * sb->numAtoms);
  assert(d_atomData != NULL);
  cudaMemcpy(d_atomData, h_atomData, ATOM_DATA_SIZE * sizeof(Real) *
    sb->numAtoms, cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sb->atomData), &d_atomData, sizeof(Real*),
    cudaMemcpyHostToDevice);

  // Copy in atomCoordinates
  cudaMalloc(&d_atomCoordinates, NUM_DIMENSIONS * sizeof(Real) * sb->numAtoms);
  assert(d_atomCoordinates != NULL);
  cudaMemcpy(d_atomCoordinates, h_atomCoordinates, NUM_DIMENSIONS *
    sizeof(Real) * sb->numAtoms, cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sb->atomCoordinates), &d_atomCoordinates, sizeof(Real*),
    cudaMemcpyHostToDevice);

  // Copy in rollBackCoordinates
  cudaMalloc(&d_rollBackCoordinates, NUM_DIMENSIONS * sizeof(Real)
    * sb->numAtoms);
  assert(d_rollBackCoordinates != NULL);
  cudaMemcpy(d_rollBackCoordinates, h_rollBackCoordinates, NUM_DIMENSIONS *
    sizeof(Real) * sb->numAtoms, cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sb->rollBackCoordinates), &d_rollBackCoordinates,
    sizeof(Real*), cudaMemcpyHostToDevice);

  // Copy in primaryIndexes
  cudaMalloc(&d_primaryIndexes, sb->numPIdxes * sizeof(int));
  assert(d_primaryIndexes != NULL);
  cudaMemcpy(d_primaryIndexes, sb->primaryIndexes, sb->numPIdxes * sizeof(int),
    cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sb->primaryIndexes), &d_primaryIndexes, sizeof(int*),
    cudaMemcpyHostToDevice);

  // Copy in box size
  cudaMalloc(&d_size, NUM_DIMENSIONS * sizeof(Real));
  assert(d_size != NULL);
  cudaMemcpy(d_size, sb->size, NUM_DIMENSIONS * sizeof(Real),
    cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sb->size), &d_size, sizeof(Real*), cudaMemcpyHostToDevice);

  // TODO - ANGLE AND BOND DATA
  cudaMalloc(&d_angleData, ANGLE_DATA_SIZE * sizeof(Real*));
  assert(d_angleData != NULL);

  Real* tmp_angleDataRows[ANGLE_DATA_SIZE];
  for (int row = 0; row < ANGLE_DATA_SIZE; row++) {
    cudaMalloc( (void**)&tmp_angleDataRows[row], sizeof(Real) * sb->numAngles);
    cudaMemcpy(tmp_angleDataRows[row], sb->angleData[row], sb->numAngles * sizeof(Real), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_angleData, tmp_angleDataRows, sizeof(tmp_angleDataRows), cudaMemcpyHostToDevice);

  cudaMalloc(&d_angleSizes, sb->numAngles * sizeof(Real));
  assert(d_angleSizes != NULL);
  cudaMemcpy(d_angleSizes, sb->angleSizes, sb->numAngles * sizeof(Real), cudaMemcpyHostToDevice);

  cudaMalloc(&d_rollBackAngleSizes, sb->numAngles * sizeof(Real));
  assert(d_rollBackAngleSizes != NULL);
  cudaMemcpy(d_rollBackAngleSizes, sb->rollBackAngleSizes, sb->numAngles * sizeof(Real), cudaMemcpyHostToDevice);


  cudaMalloc(&d_bondData, BOND_DATA_SIZE * sizeof(Real*));
  assert(d_bondData != NULL);

  Real * tmp_bondDataRows[BOND_DATA_SIZE];
  for (int row = 0; row < BOND_DATA_SIZE; row++) {
    cudaMalloc( (void**)&tmp_bondDataRows[row], sizeof(Real) * sb->numBonds);
    cudaMemcpy(tmp_bondDataRows[row], sb->bondData[row], sb->numBonds * sizeof(Real), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_bondData, tmp_bondDataRows, sizeof(tmp_bondDataRows), cudaMemcpyHostToDevice);

  cudaMalloc(&d_bondLengths, sb->numBonds * sizeof(Real));
  assert(d_bondLengths != NULL);
  cudaMemcpy(d_bondLengths, sb->bondLengths, sb->numBonds * sizeof(Real), cudaMemcpyHostToDevice);

  cudaMalloc(&d_rollBackBondLengths, sb->numBonds * sizeof(Real));
  assert(d_rollBackBondLengths != NULL);
  cudaMemcpy(d_rollBackBondLengths, sb->rollBackBondLengths, sb->numBonds * sizeof(Real), cudaMemcpyHostToDevice);
}

void GPUCopy::copyOut(SimBox* sb) {
  // Copy out moleculeData.
  cudaMemcpy(h_moleculeData, d_moleculeData, MOL_DATA_SIZE * sizeof(int) *
    sb->numMolecules, cudaMemcpyDeviceToHost);

  // Copy out atomData.
  cudaMemcpy(h_atomData, d_atomData, ATOM_DATA_SIZE * sizeof(Real) *
    sb->numAtoms, cudaMemcpyDeviceToHost);

  // Copy out atomCoordinates and rollback coordinates.
  cudaMemcpy(h_atomCoordinates, d_atomCoordinates, NUM_DIMENSIONS *
    sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_rollBackCoordinates, d_rollBackCoordinates, sb->largestMol *
    sizeof(Real), cudaMemcpyDeviceToHost);

  // Copy out primaryIndexes
  cudaMemcpy(h_primaryIndexes, d_primaryIndexes, sb->numPIdxes * sizeof(int),
    cudaMemcpyDeviceToHost);

  // Copy out box dimensions
  cudaMemcpy(h_size, d_size, NUM_DIMENSIONS * sizeof(Real),
    cudaMemcpyDeviceToHost);

  // Copy out angles and angle rollback data.
  cudaMemcpy(h_angleSizes, d_angleSizes, sb->numAngles * sizeof(Real),
    cudaMemcpyDeviceToHost);
  cudaMemcpy(h_rollBackAngleSizes, d_rollBackAngleSizes, sb->numAngles *
    sizeof(Real), cudaMemcpyDeviceToHost);

  // Copy out bonds and bond length data.
  cudaMemcpy(h_bondLengths, d_bondLengths , sb->numBonds * sizeof(Real),
    cudaMemcpyDeviceToHost);
  cudaMemcpy(h_rollBackBondLengths, d_rollBackBondLengths , sb->numBonds *
    sizeof(Real), cudaMemcpyDeviceToHost);
}

