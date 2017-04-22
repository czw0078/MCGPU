#include "GPUCopy.h"

bool launchOnGpu = false;

SimBox* h_sb = NULL;
SimBox* d_sb = NULL;

Real* d_atomData = NULL;

Real* d_rollBackCoordinates = NULL;

Real* d_atomCoordinates = NULL;

int* d_primaryIndexes = NULL;

int* d_moleculeData = NULL;

Real* d_size = NULL;

// Angle values

Real** d_angleData = NULL;

Real* d_angleSizes = NULL;

Real* d_rollBackAngleSizes = NULL;

// Bond values

Real** d_bondData = NULL;

Real* d_bondLengths = NULL;

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

Real* GPUCopy::atomCoordinatesPtr()	{
	return d_atomCoordinates;
}

int* GPUCopy::moleculeDataPtr()	{
	return d_moleculeData;
}

Real** angleDataPtr()	{
	return d_angleData;
}

Real* angleSizesPtr()	{
	return d_angleSizes;
}

Real* rollBackAngleSizesPtr()	{
	return d_rollBackAngleSizes;
}

Real** bondDataPtr()	{
	return d_bondData;
}

Real* bondLengthsPtr()	{
	return d_bondLengths;
}

Real* rollBackBondLengthsPtr()	{
	return d_rollBackBondLengths;
}

void GPUCopy::copyIn(SimBox *sb) {
  h_sb = sb;

  cudaMalloc(&d_sb, sizeof(SimBox));
  assert(d_sb != NULL);
  cudaMemcpy(d_sb, h_sb, sizeof(SimBox), cudaMemcpyHostToDevice);

  // Copy in moleculeData

  cudaMalloc(&d_moleculeData, MOL_DATA_SIZE * sizeof(int) * sb->numMolecules);
  assert(d_moleculeData != NULL);
  cudaMemcpy(d_moleculeData, sb->moleculeData, MOL_DATA_SIZE * sizeof(int) *
    sb->numMolecules, cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sb->moleculeData), &d_moleculeData, sizeof(int*),
    cudaMemcpyHostToDevice);

  // Copy in atomData
  cudaMalloc(&d_atomData, ATOM_DATA_SIZE * sizeof(Real) * sb->numAtoms);
  assert(d_atomData != NULL);
  cudaMemcpy(d_atomData, sb->atomData, ATOM_DATA_SIZE * sizeof(Real) *
    sb->numAtoms, cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sb->atomData), &d_atomData, sizeof(Real*),
    cudaMemcpyHostToDevice);

  // Copy in atomCoordinates
  cudaMalloc(&d_atomCoordinates, NUM_DIMENSIONS * sizeof(Real) * sb->numAtoms);
  assert(d_atomCoordinates != NULL);
  cudaMemcpy(d_atomCoordinates, sb->atomCoordinates, NUM_DIMENSIONS *
    sizeof(Real) * sb->numAtoms, cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sb->atomCoordinates), &d_atomCoordinates, sizeof(Real*),
    cudaMemcpyHostToDevice);
  Real* devPtr;
  cudaMemcpy(&devPtr, &(d_sb->atomCoordinates), sizeof(Real*), cudaMemcpyDeviceToHost);

  // Copy in rollBackCoordinates
  cudaMalloc(&d_rollBackCoordinates, NUM_DIMENSIONS * sizeof(Real)
    * sb->numAtoms);
  assert(d_rollBackCoordinates != NULL);
  cudaMemcpy(d_rollBackCoordinates, sb->rollBackCoordinates, NUM_DIMENSIONS *
    sizeof(Real) * sb->numAtoms, cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sb->rollBackCoordinates), &d_rollBackCoordinates,
    sizeof(Real*), cudaMemcpyHostToDevice);
  cudaMemcpy(&devPtr, &(d_sb->rollBackCoordinates), sizeof(Real*), cudaMemcpyDeviceToHost);

  // Copy in primaryIndexes
  cudaMalloc(&d_primaryIndexes, sb->numPIdxes * sizeof(int));
  assert(d_primaryIndexes != NULL);
  cudaMemcpy(d_primaryIndexes, sb->primaryIndexes, sb->numPIdxes * sizeof(int),
    cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sb->primaryIndexes), &d_primaryIndexes, sizeof(int*),
    cudaMemcpyHostToDevice);
  int* devIntPtr;
  cudaMemcpy(&devIntPtr, &(d_sb->primaryIndexes), sizeof(int*), cudaMemcpyDeviceToHost);

  // Copy in box size
  cudaMalloc(&d_size, NUM_DIMENSIONS * sizeof(Real));
  assert(d_size != NULL);
  cudaMemcpy(d_size, sb->size, NUM_DIMENSIONS * sizeof(Real),
    cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_sb->size), &d_size, sizeof(Real*), cudaMemcpyHostToDevice);
  cudaMemcpy(&devPtr, &(d_sb->size), sizeof(Real*), cudaMemcpyDeviceToHost);

  cudaMemcpy(&devPtr, &(d_sb->atomCoordinates), sizeof(Real*), cudaMemcpyDeviceToHost);

  // Copy in Angle Data
	cudaMalloc(&d_angleData, sb->numAngles * sizeof(Real) * ANGLE_DATA_SIZE);
	assert(d_angleData != NULL);
	cudaMemcpy(d_angleData, sb->angleData, sb->numAngles * sizeof(Real) * ANGLE_DATA_SIZE,
		cudaMemcpyHostToDevice);

	cudaMalloc(&d_angleSizes, sb->numAngles * sizeof(Real));
	assert(d_angleSizes != NULL);
	cudaMemcpy(d_angleSizes, sb->angleSizes, sb->numAngles * sizeof(Real), 
		cudaMemcpyHostToDevice);

	cudaMalloc(&d_rollBackAngleSizes, sb->numAngles * sizeof(Real));
	assert(d_rollBackAngleSizes != NULL);
	cudaMemcpy(d_rollBackAngleSizes, sb->rollBackAngleSizes, sb->numAngles * sizeof(Real),
		cudaMemcpyHostToDevice);

	// Copy in Bond Data
	cudaMalloc(&d_bondData, sb->numBonds * sizeof(Real) * BOND_DATA_SIZE);
	assert(d_bondData != NULL);
	cudaMemcpy(d_bondData, sb->bondData, sb->numBonds * sizeof(Real) * BOND_DATA_SIZE, 
		cudaMemcpyHostToDevice);

	cudaMalloc(&d_bondLengths, sb->numBonds * sizeof(Real));
	assert(d_bondLengths != NULL);
	cudaMemcpy(d_bondLengths, sb->bondLengths, sb->numBonds * sizeof(Real),
		cudaMemcpyHostToDevice);

	cudaMalloc(&d_rollBackBondLengths, sb->numBonds * sizeof(Real));
	assert(d_rollBackBondLengths != NULL);
	cudaMemcpy(d_rollBackBondLengths, sb->rollBackBondLengths, sb->numBonds * sizeof(Real),
		cudaMemcpyHostToDevice);
}

void GPUCopy::copyOut(SimBox* sb) {
  // Copy out moleculeData.
  cudaMemcpy(sb->moleculeData, d_moleculeData, MOL_DATA_SIZE * sizeof(int) *
    sb->numMolecules, cudaMemcpyDeviceToHost);

  // Copy out atomData.
  cudaMemcpy(sb->atomData, d_atomData, ATOM_DATA_SIZE * sizeof(Real) *
    sb->numAtoms, cudaMemcpyDeviceToHost);

  // Copy out atomCoordinates and rollback coordinates.
  cudaMemcpy(sb->atomCoordinates, d_atomCoordinates, NUM_DIMENSIONS *
    sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(sb->rollBackCoordinates, d_rollBackCoordinates, sb->largestMol *
    sizeof(Real), cudaMemcpyDeviceToHost);

  // Copy out primaryIndexes
  cudaMemcpy(sb->primaryIndexes, d_primaryIndexes, sb->numPIdxes * sizeof(int),
    cudaMemcpyDeviceToHost);

  // Copy out box dimensions
  cudaMemcpy(sb->size, d_size, NUM_DIMENSIONS * sizeof(Real),
    cudaMemcpyDeviceToHost);

  // Copy out angles and angle rollback data.
  cudaMemcpy(sb->angleSizes, d_angleSizes, sb->numAngles * sizeof(Real),
    cudaMemcpyDeviceToHost);
  cudaMemcpy(sb->rollBackAngleSizes, d_rollBackAngleSizes, sb->numAngles *
    sizeof(Real), cudaMemcpyDeviceToHost);
	//?
	cudaMemcpy(sb->angleData, d_angleData, sb->numAngles * sizeof(Real) * 
		ANGLE_DATA_SIZE, cudaMemcpyDeviceToHost);

  // Copy out bonds and bond length data.
  cudaMemcpy(sb->bondLengths, d_bondLengths , sb->numBonds * sizeof(Real),
    cudaMemcpyDeviceToHost);
  cudaMemcpy(sb->rollBackBondLengths, d_rollBackBondLengths , sb->numBonds *
    sizeof(Real), cudaMemcpyDeviceToHost);
	cudaMemcpy(sb->bondData, d_bondData, sb->numBonds * sizeof(Real) * 
		BOND_DATA_SIZE, cudaMemcpyDeviceToHost);
}

