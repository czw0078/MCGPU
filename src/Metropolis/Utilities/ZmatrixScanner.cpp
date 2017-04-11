#include "FileUtilities.h"

#include <exception>
#include <stdexcept>
//#include <sstream>
#include "Parsing.h"
#include "StructLibrary.h"
#include "Metropolis/Box.h"
#include "Metropolis/SimulationArgs.h"


ZmatrixScanner::ZmatrixScanner() {
  oplsScanner = NULL;
  startNewMolecule = false;
  previousFormat = 0;
}

ZmatrixScanner::~ZmatrixScanner() {
}

bool ZmatrixScanner::readInZmatrix(string filename, OplsScanner* scanner) {
	fileName = filename;
	oplsScanner = scanner;
	startNewMolecule = false;

	if (fileName.empty()) {
		std::cerr << "Error: readInZmatrix(): Given filename is NULL" << std::endl;
		return false;
	}

  stringstream output;
  int numOfLines=0;

  ifstream zmatrixScanner(fileName.c_str());

  if (!zmatrixScanner.is_open()) {
    std::cerr << "Error: Unable to open Z-Matrix file (" << fileName << ")" << std::endl;
    return false;
  }

  string line;

  while (zmatrixScanner.good()) {
    numOfLines++;
    getline(zmatrixScanner, line);

    try {
      if (line.at(0) != '#' && numOfLines > 1) {
        parseLine(line,numOfLines);
      }
    } catch (std::out_of_range& e) {}

    if (startNewMolecule) {
      // TODO handle case when more than one molecule is specified in z-matrix
    }
  }

  finalizeMolecules();

  zmatrixScanner.close();

  return true;
}

void ZmatrixScanner::parseLine(string line, int numOfLines) {

  string atomID, atomType, oplsA, oplsB, bondWith, bondDistance, angleWith, angleMeasure, dihedralWith, dihedralMeasure;

  stringstream ss;

  // Check if line contains correct format
  int format = checkFormat(line);

  if(format == 1) {
    // Read in strings in columns and store the data in temporary variables
    ss << line;
    ss >> atomID >> atomType >> oplsA >> oplsB >> bondWith >> bondDistance
       >> angleWith >> angleMeasure >> dihedralWith >> dihedralMeasure;

	  // Setup structures for permanent encapsulation
    Atom lineAtom;
    Bond lineBond;
    Angle lineAngle;
    Dihedral lineDihedral;

    if (oplsA.compare("-1") != 0) {
      lineAtom = oplsScanner->getAtom(oplsA);
      lineAtom.id = atoi(atomID.c_str());
      lineAtom.x = 0;
      lineAtom.y = 0;
      lineAtom.z = 0;
    } else {
      // Dummy atom, only used for geometry purposes
      lineAtom = createAtom(atoi(atomID.c_str()), -1, -1, -1, -1, -1, 0, atomType);
    }

	  atomVector.push_back(lineAtom);

    if (bondWith.compare("0") != 0) {
      lineBond.atom1 = lineAtom.id;
      lineBond.atom2 = atoi(bondWith.c_str());
      lineBond.distance = atof(bondDistance.c_str());
      lineBond.variable = false;
      bondVector.push_back(lineBond);
    }

    if (angleWith.compare("0") != 0) {
      lineAngle.atom1 = lineAtom.id;
      lineAngle.atom2 = atoi(angleWith.c_str());
      lineAngle.value = atof(angleMeasure.c_str());
      lineAngle.commonAtom = 0;
      lineAngle.variable = false;
      angleVector.push_back(lineAngle);
    }

    if (dihedralWith.compare("0") != 0) {
      lineDihedral.atom1 = lineAtom.id;
      lineDihedral.atom2 = atoi(dihedralWith.c_str());
      lineDihedral.atom3 = atoi(bondWith.c_str());
      lineDihedral.atom4 = atoi(angleWith.c_str());
      lineDihedral.value = atof(dihedralMeasure.c_str());
      lineDihedral.variable = false;
      // We do not set Fourier coefficients here because we do not yet have enough data
      dihedralVector.push_back(lineDihedral);
    }

  } else if (format == 2) {
    startNewMolecule = true;
  } else if (format == 3) {
    startNewMolecule = true;
  }

  if (previousFormat >= 3 && format == -1) {
    handleZAdditions(line, previousFormat);
  } else {
    previousFormat = format;
  }

}

int ZmatrixScanner::checkFormat(string line) {
  int format = -1;
  stringstream iss(line);
  stringstream iss2(line);
  string atomType, someLine, extra;
  int atomID, oplsA, oplsB, bondWith, angleWith,dihedralWith;
  Real bondDistance, angleMeasure, dihedralMeasure;

    // check if it is the normal 11 line format
  if (iss >> atomID >> atomType >> oplsA >> oplsB >> bondWith >> bondDistance >>
        angleWith >> angleMeasure >> dihedralWith >> dihedralMeasure >> extra) {

    format = 1;

  } else {
    someLine = line;

    if(someLine.find("TERZ")!=string::npos) {
      format = 2;
		} else if(someLine.find("Geometry Variations")!=string::npos) {
      format = 3;
    } else if(someLine.find("Variable Bonds")!=string::npos) {
      format = 4;
    } else if(someLine.find("Additional Bonds")!=string::npos) {
      format = 5;
    } else if(someLine.find("Harmonic Constraints")!=string::npos) {
      format = 6;
    } else if(someLine.find("Variable Bond Angles")!=string::npos) {
      format = 7;
    } else if(someLine.find("Additional Bond Angles")!=string::npos) {
      format = 8;
    } else if(someLine.find("Variable Dihedrals")!=string::npos) {
      format = 9;
    } else if(someLine.find("Additional Dihedrals")!=string::npos) {
      format = 10;
    } else if(someLine.find("Domain Definitions")!=string::npos) {
      format = 11;
    } else if(someLine.find("Final blank line")!=string::npos) {
      format = -2;
    }
  }
    return format;
}

void ZmatrixScanner::handleZAdditions(string line, int cmdFormat) {
  if (line.find("AUTO") != string::npos) {
	     //Do stuff for AUTO
	     //but what is AUTO-related stuff?
  } else {
    switch(cmdFormat) {
      case 3:     // Geometry Variations follow
        break;
      case 4:     // Variable Bonds follow
        handleVariableBond(line);
        break;
      case 5:     //  Additional Bonds follow
        handleAdditionalBond(line);
        break;
      case 6:     // Harmonic Constraints follow
        break;
      case 7:     //  Variable Bond Angles follow
        handleVariableAngle(line);
        break;
      case 8:     // Additional Bond Angles follow
        handleAdditionalAngle(line);
        break;
      case 9:     // Variable Dihedrals follow
        handleVariableDihedral(line);
        break;
      case 10:    //  Additional Dihedrals follow
        handleAdditionalDihedral(line);
        break;
    }
  }
}

vector<Hop> ZmatrixScanner::calculateHops(Molecule molec)
{
    vector<Hop> newHops;
    int **graph;
    int size = molec.numOfAtoms;
	int startId = molec.atoms[0].id;

    buildAdjacencyMatrix(graph,molec);

    for(int atom1=0; atom1<size; atom1++)
    {
        for(int atom2=atom1+1; atom2<size; atom2++)
        {
            int distance = findHopDistance(atom1,atom2,size,graph);
            if(distance >=3)
            {
				Hop tempHop = Hop(atom1+startId,atom2+startId,distance); //+startId because atoms may not start at 1
                newHops.push_back(tempHop);
            }
        }
    }

    return newHops;
}

bool ZmatrixScanner::contains(vector<int> &vect, int item)
{
    for(int i=0; i<vect.size(); i++)
    {
        if(vect[i]==item)
        {
            return true;
        }
    }

    return false;
}


int ZmatrixScanner::findHopDistance(int atom1, int atom2, int size, int **graph) {
  map<int, int> distance;
  queue<int> Queue;
  vector<int> checked;
  vector<int> bonds;

  Queue.push(atom1);
  checked.push_back(atom1);
  distance.insert(pair<int, int>(atom1, 0));

  while (!Queue.empty()) {
    int target = Queue.front();
    Queue.pop();

    if(target == atom2) {
      return distance[target];
    }

    bonds.clear();

    for (int col = 0; col < size; col++) {
      if (graph[target][col] == 1) {
        bonds.push_back(col);
      }
    }

    for (int x = 0; x < bonds.size(); x++) {
      int currentBond = bonds[x];

      if (!contains(checked, currentBond)) {
        checked.push_back(currentBond);
        int newDistance = distance[target] + 1;
        distance.insert(pair<int, int>(currentBond, newDistance));
        Queue.push(currentBond);
      }
    }
  }

  //Needs a return value
  return -1; //Temp fill
}

void ZmatrixScanner::buildAdjacencyMatrix(int **&graph, Molecule molec) {

  int size = molec.numOfAtoms;
	int startId = molec.atoms[0].id;
	int lastId = startId + molec.numOfAtoms -1;

  graph =  new int*[size]; //create colums

  for (int i = 0; i < size; i++) { //create rows
    graph[i] = new int[size];
  }

  //fill with zero
  for (int c = 0; c < size; c++) {
    for (int r = 0; r < size; r++) {
            graph[c][r]=0;
    }
  }

  //fill with adjacent array with bonds
  for(int x=0; x<molec.numOfBonds; x++) {
    Bond bond = molec.bonds[x];
		//make sure the bond is intermolecular
		if ( (bond.atom1 >= startId && bond.atom1 <= lastId) &&
		      (bond.atom2 >= startId && bond.atom2 <= lastId) ) {

      graph[bond.atom1-startId][bond.atom2-startId] = 1;
      graph[bond.atom2-startId][bond.atom1-startId] = 1;
		}
  }
}

vector<Molecule> ZmatrixScanner::buildMolecule(int startingID) {
	int numOfMolec = moleculePattern.size();
	Molecule newMolecules[numOfMolec];

    //need a deep copy of molecule pattern incase it is modified.
  for (int i = 0; i < moleculePattern.size(); i++) {
    Atom *atomCopy = new Atom[ moleculePattern[i].numOfAtoms] ;

    for (int a = 0; a < moleculePattern[i].numOfAtoms ; a++) {
      atomCopy[a] = moleculePattern[i].atoms[a];
    }

    Bond *bondCopy = new Bond[moleculePattern[i].numOfBonds];
    for (int a = 0; a < moleculePattern[i].numOfBonds; a++) {
      bondCopy[a]=  moleculePattern[i].bonds[a];
    }

    Angle *angleCopy = new Angle[moleculePattern[i].numOfAngles];
    for (int a = 0; a < moleculePattern[i].numOfAngles; a++) {
      angleCopy[a]=  moleculePattern[i].angles[a];
    }

    Dihedral *dihedCopy = new Dihedral[moleculePattern[i].numOfDihedrals];
    for (int a = 0; a < moleculePattern[i].numOfDihedrals; a++) {
      dihedCopy[a] = moleculePattern[i].dihedrals[a];
    }

    // Calculate and add array of Hops to the molecule
    vector<Hop> calculatedHops;
    calculatedHops = calculateHops(moleculePattern[i]);
    int numOfHops = calculatedHops.size();
    Hop *hopCopy = new Hop[numOfHops];

    for (int a = 0; a < numOfHops; a++) {
      hopCopy[a] = calculatedHops[a];
    }

    Molecule molecCopy = Molecule(-1, moleculePattern[i].type, atomCopy,
                                  angleCopy, bondCopy, dihedCopy, hopCopy,
                                  moleculePattern[i].numOfAtoms,
                                  moleculePattern[i].numOfAngles,
                                  moleculePattern[i].numOfBonds,
                                  moleculePattern[i].numOfDihedrals,
                                  numOfHops);

    newMolecules[i] = molecCopy;
  }

	//Assign/calculate the appropiate x,y,z positions to the molecules.
	//buildMoleculeInSpace(newMolecules, numOfMolec);
	buildMoleculeXYZ(newMolecules, numOfMolec);


  for (int i = 0; i < numOfMolec; i++) {
    if(i == 0) {
      newMolecules[i].id = startingID;
    } else {
      newMolecules[i].id = newMolecules[i-1].id + newMolecules[i-1].numOfAtoms;
    }
  }

  for (int j = 0; j < numOfMolec; j++) {
    Molecule newMolecule = newMolecules[j];
    //map unique IDs to atoms within structs based on startingID

    for (int i = 0; i < newMolecules[j].numOfAtoms; i++) {
      int atomID = newMolecule.atoms[i].id - 1;
      newMolecule.atoms[i].id = atomID + startingID;
    }

    for (int i = 0; i < newMolecule.numOfBonds; i++) {
      int atom1ID = newMolecule.bonds[i].atom1 - 1;
      int atom2ID = newMolecule.bonds[i].atom2 - 1;

			newMolecule.bonds[i].atom1 = atom1ID + startingID;
      newMolecule.bonds[i].atom2 = atom2ID + startingID;
    }

    for (int i = 0; i < newMolecule.numOfAngles; i++) {
      int atom1ID = newMolecule.angles[i].atom1 - 1;
      int atom2ID = newMolecule.angles[i].atom2 - 1;

			newMolecule.angles[i].atom1 = atom1ID + startingID;
      newMolecule.angles[i].atom2 = atom2ID + startingID;
    }

    for (int i = 0; i < newMolecule.numOfDihedrals; i++) {
      int atom1ID = newMolecule.dihedrals[i].atom1 - 1;
      int atom2ID = newMolecule.dihedrals[i].atom2 - 1;
      int atom3ID = newMolecule.dihedrals[i].atom3 - 1;
      int atom4ID = newMolecule.dihedrals[i].atom4 - 1;

      newMolecule.dihedrals[i].atom1 = atom1ID + startingID;
      newMolecule.dihedrals[i].atom2 = atom2ID + startingID;
      newMolecule.dihedrals[i].atom3 = atom3ID + startingID;
      newMolecule.dihedrals[i].atom4 = atom4ID + startingID;
    }

    for (int i = 0; i < newMolecule.numOfHops; i++) {
      int atom1ID = newMolecule.hops[i].atom1 - 1;
      int atom2ID = newMolecule.hops[i].atom2 - 1;

	    newMolecule.hops[i].atom1 = atom1ID + startingID;
      newMolecule.hops[i].atom2 = atom2ID + startingID;
    }
  }

  return vector<Molecule>(newMolecules,newMolecules+sizeof(newMolecules)/sizeof(Molecule));
}

void ZmatrixScanner::addImpliedAngles(vector<Angle>& angleVector,
                                      vector<Bond> bondVector) {
  // Add the midpoints (common atoms) for each angle
  for (int i = 0; i < angleVector.size(); i++) {
    int a1 = angleVector[i].atom1, a2 = angleVector[i].atom2;
    angleVector[i].commonAtom = getCommonAtom(bondVector, a1, a2);
  }

  // An angle is implied if two bonds share a midpoint and one end
  // i.e. 5-3-1 and 4-3-1 implies 5-3-4
  vector<Angle> toAdd;
  for (int i = 0; i < angleVector.size(); i++) {
    for (int j = i + 1; j < angleVector.size(); j++) {
      if (angleVector[i].commonAtom == angleVector[j].commonAtom) {
        Angle angle1 = angleVector[i], angle2 = angleVector[j];
        int a1 = 0, a2 = 0;

        // Locate the similar ends
        if (angle1.atom1 == angle2.atom1) {
          a1 = angle1.atom2, a2 = angle2.atom2;
        } else if (angle1.atom1 == angle2.atom2) {
          a1 = angle1.atom2, a2 = angle2.atom1;
        } else if (angle1.atom2 == angle2.atom1) {
          a1 = angle1.atom1, a2 = angle2.atom2;
        } else if (angle1.atom2 == angle2.atom2) {
          a1 = angle1.atom1, a2 = angle2.atom1;
        }

        // Create the new bond if we found one
        if (a1 != 0 && a2 != 0) {
          Angle newAngle;
          newAngle.atom1 = a1;
          newAngle.atom2 = a2;
          newAngle.commonAtom = getCommonAtom(bondVector, newAngle.atom1,
                                              newAngle.atom2);
          newAngle.value = 0; // To be calculated with xyz coordinates
          newAngle.variable = angle1.variable && angle2.variable; // FIXME: Not sure if that's correct

          // Ensure we're not duplicating any angles
          bool add = true;
          for (int angle = 0; angle < angleVector.size() && add; angle++) {
            if (newAngle.commonAtom == angleVector[angle].commonAtom) {
              Angle current = angleVector[angle];
              // Angle is considered equal if first, mid, and second atoms
              // are the same, regardless of exact ordering
              if ((current.atom1 == a1 && current.atom2 == a2) ||
                  (current.atom1 == a2 && current.atom2 == a1)) {
                add = false;
              }
            }
          }
          if (add) {
            toAdd.push_back(newAngle);
          }
        }
      }
    }
  }

  // Add the implied angles to our current list
  for (int i = 0; i < toAdd.size(); i++) {
    angleVector.push_back(toAdd[i]);
  }
}

void ZmatrixScanner::finalizeMolecules() {
  Atom* atomArray;
  Bond* bondArray;
  Angle* angleArray;
  Dihedral* dihedralArray;
  int moleculeNum = 0;

  // FIXME: should this be called here, or in handleZAdditions()?
  if (hasAdditionalBondAngles) {
    addImpliedAngles(angleVector, bondVector);
  }

  // Determine whether each dihedral is an "improper" torsion
  for (int i = 0; i < dihedralVector.size(); i++) {
    dihedralVector[i].isProper = isProperTorsion(dihedralVector[i]);
  }

  double atomVectorSize = atomVector.size();
  atomArray = (Atom*) malloc(sizeof(Atom) * atomVector.size());
  bondArray = (Bond*) malloc(sizeof(Bond) * bondVector.size());
  angleArray = (Angle*) malloc(sizeof(Angle) * angleVector.size());
  dihedralArray = (Dihedral*) malloc(sizeof(Dihedral) * dihedralVector.size());

  for (int i = 0; i < atomVector.size(); i++) {
    atomArray[i] = atomVector[i];
  }

  for (int i = 0; i < bondVector.size(); i++) {
    bondArray[i] = bondVector[i];
  }

  for (int i = 0; i < angleVector.size(); i++) {
    angleArray[i] = angleVector[i];
  }

  for (int i = 0; i < dihedralVector.size(); i++) {
    dihedralArray[i] = dihedralVector[i];
  }

  moleculePattern.push_back(createMolecule(-1, moleculeNum, atomArray, angleArray, bondArray, dihedralArray,
  atomVector.size(), angleVector.size(), bondVector.size(), dihedralVector.size()));

  atomVector.clear();
  bondVector.clear();
  angleVector.clear();
  dihedralVector.clear();

  startNewMolecule = false;
  hasAdditionalBondAngles = false;
}

void ZmatrixScanner::handleVariableBond(string line) {
  vector<string> tokens = tokenizeAdditionalLine(line);
  if (tokens.size() == 1) {
    int atom1 = atoi(tokens[0].c_str());
    for (int i = 0; i < bondVector.size(); i++) {
      if (bondVector[i].atom1 == atom1) {
        bondVector[i].variable = true;
      }
    }
  } else {
    // TODO Handle cases where more than one bond is specified on a line
  }
}

void ZmatrixScanner::handleAdditionalBond(string line) {
  // TODO implement this. Will need an example of a z-matrix with a ring for testing
}

void ZmatrixScanner::handleVariableAngle(string line) {
  vector<string> tokens = tokenizeAdditionalLine(line);
  if (tokens.size() == 1) {
    int atom1 = atoi(tokens[0].c_str());
    for (int i = 0; i < angleVector.size(); i++) {
      if (angleVector[i].atom1 == atom1) {
        angleVector[i].variable = true;
      }
    }
  } else {
    // TODO Handle cases where more than one angle is specified on a line
  }
}

void ZmatrixScanner::handleAdditionalAngle(string line) {
  vector<string> tokens = tokenizeAdditionalLine(line);
  if (tokens.size() == 3) {
    int atom1 = atoi(tokens[0].c_str());
    int mid = atoi(tokens[1].c_str());
    int atom2 = atoi(tokens[2].c_str());
    angleVector.push_back(Angle(atom1, atom2, 0, false));
  }
}

void ZmatrixScanner::handleVariableDihedral(string line) {
  vector<string> tokens = tokenizeAdditionalLine(line);
  if (tokens.size() == 4) {
    int atom1 = atoi(tokens[0].c_str());
    int initialType = atoi(tokens[1].c_str());
    // int finalType = atoi(tokens[2].c_str());
    Real maxAngleChange = atof(tokens[3].c_str());

    char hashNumString[10];
    sprintf(hashNumString, "%03d", initialType);
    Fourier fourierData = {0,0,0,0};
    if (initialType != 0) {
     fourierData = oplsScanner->getFourier(string(hashNumString));
    }
    //TODO what to do if fourierData is not found in opls.par?

    for (int i = 0; i < dihedralVector.size(); i++) {
      if (dihedralVector[i].atom1 == atom1) {
        dihedralVector[i].V1 = fourierData.vValues[0];
        dihedralVector[i].V2 = fourierData.vValues[1];
        dihedralVector[i].V3 = fourierData.vValues[2];
        dihedralVector[i].V4 = fourierData.vValues[3];
        dihedralVector[i].minAngleMeasure = dihedralVector[i].value - maxAngleChange;
        dihedralVector[i].maxAngleMeasure = dihedralVector[i].value + maxAngleChange;
        dihedralVector[i].variable = true;
      }
    }
  } else {
    // FIXME: What do we do if the line doesn't have format %d %d %d %f?
  }

}

void ZmatrixScanner::handleAdditionalDihedral(string line) {
  vector<string> tokens = tokenizeAdditionalLine(line);
  if (tokens.size() == 6) {
    Dihedral newDihedral;
    newDihedral.atom1 = atoi(tokens[0].c_str());
    newDihedral.atom3 = atoi(tokens[1].c_str());
    newDihedral.atom4 = atoi(tokens[2].c_str());
    newDihedral.atom2 = atoi(tokens[3].c_str());
    int initialType = atoi(tokens[4].c_str());
    // int finalType = atoi(tokens[5].c_str());
    char hashNumString[10];
    sprintf(hashNumString, "%03d", initialType);
    Fourier fourierData = {0,0,0,0};
    if (initialType != 0) {
     fourierData = oplsScanner->getFourier(string(hashNumString));
    }
    //TODO what to do if fourierData is not found in opls.par?

    newDihedral.V1 = fourierData.vValues[0];
    newDihedral.V2 = fourierData.vValues[1];
    newDihedral.V3 = fourierData.vValues[2];
    newDihedral.V4 = fourierData.vValues[3];
    newDihedral.value = 0;
    newDihedral.minAngleMeasure = 0;
    newDihedral.maxAngleMeasure = 0;
    newDihedral.variable = false;
    newDihedral.isProper = true;

    dihedralVector.push_back(newDihedral);
  } else {
    //TODO what if there are not exactly 6 arguments?
  }
}

vector<string> ZmatrixScanner::tokenizeAdditionalLine(string line) {
  vector<string> tokens;
  const char *delims = " ,-";

  char *line_c = strdup(line.c_str());
  char *result = strtok(line_c, delims);
  while (result != NULL) {
    tokens.push_back(string(result));
    result = strtok(NULL, delims);
  }
  free(line_c);
  return tokens;
}

bool ZmatrixScanner::isProperTorsion(Dihedral dih) {
  //TODO if dih.atom2 is bonded to dih.atom4, return true. Else, return false.
  for (int i = 0; i < bondVector.size(); i++) {
    Bond b = bondVector[i];
    if (b.atom1 == dih.atom2 && b.atom2 == dih.atom4 ||
        b.atom2 == dih.atom2 && b.atom1 == dih.atom4) {
      return true;
    }
  }
  return false;
}
