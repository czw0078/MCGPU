#ifndef GPUCOPY_H
#define GPUCOPY_H

#include "SimBox.h"

namespace GPUCopy {
  void copyIn(SimBox* sb);
  void copyOut(SimBox* sb);
  void setParallel(bool in);

  Real** atomDataPtr();

  Real** rollBackCoordinatesPtr();
  Real** atomCoordinatesPtr();

  Real** bondDataPtr();
  Real* bondLengthsPtr();
  Real* rollBackBondsPtr();

  Real** angleDataPtr();
  Real* angleSizesPtr();
  Real* rollBackAnglesPtr();

  int* primaryIndexesPtr();
  int* moleculeDataPtr();
  Real* sizePtr();
  int onGpu();

  __host__ __device__
  int getNumMolecules();
}


#endif
