#ifdef _OPENACC
#include <openacc.h>
#endif

#include "GPUCopy.h"

bool parallel = false;

// Scalars

int numMolecules = 0;

Real** h_atomData = NULL;
Real** d_atomData = NULL;

Real** h_rollBackCoordinates = NULL;
Real** d_rollBackCoordinates = NULL;

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

void GPUCopy::setParallel(bool in) { parallel = in; }

int GPUCopy::onGpu() { return parallel; }

Real** GPUCopy::atomDataPtr() { return parallel ? d_atomData : h_atomData; }

Real** GPUCopy::rollBackCoordinatesPtr() {
  return parallel ? d_rollBackCoordinates : h_rollBackCoordinates;
}

Real** GPUCopy::atomCoordinatesPtr() {
  return parallel ? d_atomCoordinates : h_atomCoordinates;
}

int* GPUCopy::primaryIndexesPtr() {
  return parallel ? d_primaryIndexes : h_primaryIndexes;
}

int* GPUCopy::moleculeDataPtr() {
  return parallel ? d_moleculeData : h_moleculeData;
}

Real** GPUCopy::bondDataPtr() {
  return parallel ? d_bondData : h_bondData;
}

Real* GPUCopy::bondLengthsPtr() {
  return parallel ? d_bondLengths : h_bondLengths;
}

Real* GPUCopy::rollBackBondsPtr() {
  return parallel ? d_rollBackBondLengths : h_rollBackBondLengths;
}

Real** GPUCopy::angleDataPtr() {
  return parallel ? d_angleData : h_angleData;
}

Real* GPUCopy::angleSizesPtr() {
  return parallel ? d_angleSizes : h_angleSizes;
}

Real* GPUCopy::rollBackAnglesPtr() {
  return parallel ? d_rollBackAngleSizes : h_rollBackAngleSizes;
}

Real* GPUCopy::sizePtr() { return parallel ? d_size : h_size; }

int GPUCopy::getNumMolecules() { return numMolecules; }

void GPUCopy::copyIn(SimBox *sb) {
  numMolecules = sb->numMolecules;
  h_moleculeData = sb->moleculeData;
  h_atomData = sb->atomData;
  h_atomCoordinates = sb->atomCoordinates;
  h_rollBackCoordinates = sb->rollBackCoordinates;
  h_size = sb->size;
  h_primaryIndexes = sb->primaryIndexes;
  h_bondData = sb->bondData;
  h_bondLengths = sb->bondLengths;
  h_rollBackBondLengths = sb->rollBackBondLengths;
  h_angleData = sb->angleData;
  h_angleSizes = sb->angleSizes;
  h_rollBackAngleSizes = sb->rollBackAngleSizes;
  if (!parallel) { return; }

  // Copy in moleculeData
  cudaMalloc(&d_moleculeData, MOL_DATA_SIZE * sizeof(int) * sb->numMolecules);
  assert(d_moleculeData != NULL);
  cudaMemcpy(d_moleculeData, h_moleculeData, MOL_DATA_SIZE * sizeof(int) *
    sb->numMolecules, cudaMemcpyHostToDevice);


  cudaMalloc(&d_atomData, ATOM_DATA_SIZE * sizeof(Real*));
  assert(d_atomData != NULL);

  Real* tmp_atomDataRows[ATOM_DATA_SIZE];
  for (int row = 0; row < ATOM_DATA_SIZE; row++) {
    cudaMalloc( (void**)&tmp_atomDataRows[row], sizeof(Real) * sb->numAtoms);
    cudaMemcpy(tmp_atomDataRows[row], sb->atomData[row], sizeof(Real)* sb->numAtoms,
      cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_atomData, tmp_atomDataRows, sizeof(tmp_atomDataRows), cudaMemcpyHostToDevice);

  cudaMalloc(&d_atomCoordinates, NUM_DIMENSIONS * sizeof(Real *));
  assert(d_atomCoordinates != NULL);

  Real* tmp_atomCoordinateRows[NUM_DIMENSIONS];
  for (int row = 0; row < NUM_DIMENSIONS; row++) {
    cudaMalloc( (void**)&tmp_atomCoordinateRows[row], sizeof(Real) * sb->numAtoms);
    cudaMemcpy(tmp_atomCoordinateRows[row], sb->atomCoordinates[row], sizeof(Real) * sb->numAtoms,
      cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_atomCoordinates, tmp_atomCoordinateRows, sizeof(tmp_atomCoordinateRows), cudaMemcpyHostToDevice);

  cudaMalloc(&d_rollBackCoordinates, NUM_DIMENSIONS * sizeof(Real *));
  assert(d_rollBackCoordinates != NULL);

  Real* tmp_rollBackCoordinateRows[NUM_DIMENSIONS];
  for (int row = 0; row < NUM_DIMENSIONS; row++) {
    cudaMalloc( (void**)&tmp_rollBackCoordinateRows[row], sizeof(Real) * sb->largestMol); 
    cudaMemcpy(tmp_rollBackCoordinateRows[row], sb->rollBackCoordinates[row], sizeof(Real) * sb->largestMol,
      cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_rollBackCoordinates, tmp_rollBackCoordinateRows, NUM_DIMENSIONS * sizeof(Real *), cudaMemcpyHostToDevice);

  cudaMalloc(&d_primaryIndexes, sb->numPIdxes * sizeof(int));
  cudaMemcpy(d_primaryIndexes, sb->primaryIndexes, sb->numPIdxes * sizeof(int), cudaMemcpyHostToDevice);


  cudaMalloc(&d_size, NUM_DIMENSIONS * sizeof(Real));
  cudaMemcpy(d_size, sb->size, NUM_DIMENSIONS * sizeof(Real), cudaMemcpyHostToDevice);

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
  if (!parallel) return;

  // Copy out moleculeData.
  cudaMemcpy(h_moleculeData, d_moleculeData, MOL_DATA_SIZE * sizeof(int) *
    sb->numMolecules, cudaMemcpyDeviceToHost);

  Real *tmp_atomDataRows[ATOM_DATA_SIZE];
  cudaMemcpy(tmp_atomDataRows, d_atomData, MOL_DATA_SIZE * sizeof(Real*), cudaMemcpyDeviceToHost);
  for (int row = 0; row < ATOM_DATA_SIZE; row++) {
    cudaMemcpy(h_atomData[row], tmp_atomDataRows[row], sb->numAtoms * sizeof(Real), cudaMemcpyDeviceToHost);
  }

  Real *tmp_atomCoordinateRows[NUM_DIMENSIONS];
  cudaMemcpy(tmp_atomCoordinateRows, d_atomCoordinates, NUM_DIMENSIONS * sizeof(Real*), cudaMemcpyDeviceToHost);
  for (int row = 0; row < NUM_DIMENSIONS; row++) {
    cudaMemcpy(h_atomCoordinates[row], tmp_atomCoordinateRows[row], sb->numAtoms * sizeof(Real), cudaMemcpyDeviceToHost);
  }

  Real *tmp_rollBackCoordinateRows[NUM_DIMENSIONS];
  cudaMemcpy(tmp_rollBackCoordinateRows, d_rollBackCoordinates, NUM_DIMENSIONS * sizeof(Real*), cudaMemcpyDeviceToHost);
  for (int row = 0; row < NUM_DIMENSIONS; row++) {
    cudaMemcpy(h_rollBackCoordinates[row], tmp_rollBackCoordinateRows[row], sb->largestMol * sizeof(Real), cudaMemcpyDeviceToHost);
  }

  cudaMemcpy(h_primaryIndexes, d_primaryIndexes, sb->numPIdxes * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_size, d_size, NUM_DIMENSIONS * sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_angleSizes, d_angleSizes, sb->numAngles * sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_rollBackAngleSizes, d_rollBackAngleSizes, sb->numAngles * sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_bondLengths, d_bondLengths , sb->numBonds* sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_rollBackBondLengths, d_rollBackBondLengths , sb->numBonds* sizeof(Real), cudaMemcpyDeviceToHost);
}

