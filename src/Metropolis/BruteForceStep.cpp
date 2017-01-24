#include "BruteForceStep.h"
#include "SimulationStep.h"
#include "GPUCopy.h"


Real BruteForceStep::calcMolecularEnergyContribution(int currMol,
                                                     int startMol) {
  return BruteForceCalcs::calcMolecularEnergyContribution(currMol, startMol);
}


// ----- BruteForceCalcs Definitions -----


Real BruteForceCalcs::calcMolecularEnergyContribution(int currMol,
                                                      int startMol) {
  Real total = 0;

  int* molData = GPUCopy::moleculeDataPtr();
  Real** atomCoords = GPUCopy::atomCoordinatesPtr();
  Real* bSize = GPUCopy::sizePtr();
  int* pIdxes = GPUCopy::primaryIndexesPtr();
  Real* aData = GPUCopy::atomDataPtr();
  Real cutoff = SimCalcs::sb->cutoff;
  const long numMolecules = SimCalcs::sb->numMolecules;

  const int p1Start = SimCalcs::sb->moleculeData[MOL_PIDX_START * numMolecules + currMol];
  const int p1End = (SimCalcs::sb->moleculeData[MOL_PIDX_COUNT * numMolecules + currMol]
                     + p1Start);

  for (int otherMol = startMol; otherMol < numMolecules; otherMol++) {
    if (otherMol != currMol) {
      int p2Start = molData[MOL_PIDX_START * numMolecules + otherMol];
      int p2End = molData[MOL_PIDX_COUNT * numMolecules + otherMol] + p2Start;
      if (SimCalcs::moleculesInRange(p1Start, p1End, p2Start, p2End,
                                     atomCoords, bSize, pIdxes, cutoff)) {
        total += calcMoleculeInteractionEnergy(currMol, otherMol, molData,
                                               aData, atomCoords, bSize);
      }
    }
  }

  return total;
}

Real BruteForceCalcs::calcMoleculeInteractionEnergy (int m1, int m2,
                                                     int* molData,
                                                     Real* aData,
                                                     Real** aCoords,
                                                     Real* bSize) {
  Real energySum = 0;


  const long numMolecules = SimCalcs::sb->numMolecules;
  const int numAtoms = SimCalcs::sb->numAtoms;
  const int m1Start = molData[MOL_START * numMolecules + m1];
  const int m1End = molData[MOL_LEN * numMolecules + m1] + m1Start;

  const int m2Start = molData[MOL_START * numMolecules + m2];
  const int m2End = molData[MOL_LEN * numMolecules + m2] + m2Start;

  for (int i = m1Start; i < m1End; i++) {
    for (int j = m2Start; j < m2End; j++) {
      if (aData[ATOM_SIGMA * numAtoms +  i] >= 0 && aData[ATOM_SIGMA * numAtoms + j] >= 0
          && aData[ATOM_EPSILON * numAtoms + i] >= 0 && aData[ATOM_EPSILON * numAtoms + j] >= 0) {

        const Real r2 = SimCalcs::calcAtomDistSquared(i, j, aCoords, bSize);
        if (r2 == 0.0) {
          energySum += 0.0;
        } else {
          energySum += SimCalcs::calcLJEnergy(i, j, r2, aData, numAtoms);
          energySum += SimCalcs::calcChargeEnergy(i, j, sqrt(r2), aData, numAtoms);
        }
      }
    }
  }

  return (energySum);
}
