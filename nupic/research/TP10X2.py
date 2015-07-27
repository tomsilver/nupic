#! /usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import csv
import os
import pkg_resources

import numpy
from numpy import *

import nupic.math
from nupic.research.TP import TP

from nupic.bindings.algorithms import Cells4


# Default verbosity while running unit tests
VERBOSITY = 0

# The numpy equivalent to the floating point type used by NTA
dtype = nupic.math.GetNTAReal()



def _extractCallingMethodArgs():
  """
  Returns args dictionary from the calling method
  """
  import inspect
  import copy

  callingFrame = inspect.stack()[1][0]

  argNames, _, _, frameLocalVarDict = inspect.getargvalues(callingFrame)

  argNames.remove("self")

  args = copy.copy(frameLocalVarDict)


  for varName in frameLocalVarDict:
    if varName not in argNames:
      args.pop(varName)

  return args



class TP10X2(TP):
  """Class implementing the temporal pooler algorithm as described in the
  published Cortical Learning Algorithm documentation.  The implementation here
  attempts to closely match the pseudocode in the documentation. This
  implementation does contain several additional bells and whistles such as
  a column confidence measure.
  """


  # We use the same keyword arguments as TP()
  def __init__(self,
               numberOfCols = 500,
               cellsPerColumn = 10,
               initialPerm = 0.11, # TODO: check perm numbers with Ron
               connectedPerm = 0.50,
               minThreshold = 8,
               newSynapseCount = 15,
               permanenceInc = 0.10,
               permanenceDec = 0.10,
               permanenceMax = 1.0, # never exceed this value
               globalDecay = 0.10,
               activationThreshold = 12, # 3/4 of newSynapseCount TODO make fraction
               doPooling = False, # allows to turn off pooling
               segUpdateValidDuration = 5,
               burnIn = 2,             # Used for evaluating the prediction score
               collectStats = False,    # If true, collect training and inference stats
               seed = 42,
               verbosity = VERBOSITY,
               checkSynapseConsistency = False,

               pamLength = 1,
               maxInfBacktrack = 10,
               maxLrnBacktrack = 5,
               maxAge = 100000,
               maxSeqLength = 32,

               # Fixed size mode params
               maxSegmentsPerCell = -1,
               maxSynapsesPerSegment = -1,

               # Output control
               outputType = 'normal',
               ):

    #---------------------------------------------------------------------------------
    # Save our __init__ args for debugging
    self._initArgsDict = _extractCallingMethodArgs()

    #---------------------------------------------------------------------------------
    # These two variables are for testing

    # If set to True, Cells4 will perform (time consuming) invariance checks
    self.checkSynapseConsistency = checkSynapseConsistency

    # If set to False, Cells4 will *not* be treated as an ephemeral member
    # and full TP10X pickling is possible. This is useful for testing
    # pickle/unpickle without saving Cells4 to an external file
    self.makeCells4Ephemeral = True

    #---------------------------------------------------------------------------------
    # Init the base class
    TP.__init__(self,
               numberOfCols = numberOfCols,
               cellsPerColumn = cellsPerColumn,
               initialPerm = initialPerm,
               connectedPerm = connectedPerm,
               minThreshold = minThreshold,
               newSynapseCount = newSynapseCount,
               permanenceInc = permanenceInc,
               permanenceDec = permanenceDec,
               permanenceMax = permanenceMax, # never exceed this value
               globalDecay = globalDecay,
               activationThreshold = activationThreshold,
               doPooling = doPooling,
               segUpdateValidDuration = segUpdateValidDuration,
               burnIn = burnIn,
               collectStats = collectStats,
               seed = seed,
               verbosity = verbosity,
               pamLength = pamLength,
               maxInfBacktrack = maxInfBacktrack,
               maxLrnBacktrack = maxLrnBacktrack,
               maxAge = maxAge,
               maxSeqLength = maxSeqLength,
               maxSegmentsPerCell = maxSegmentsPerCell,
               maxSynapsesPerSegment = maxSynapsesPerSegment,
               outputType = outputType,
               )

  def __setstate__(self, state):
    """
    Set the state of ourself from a serialized state.
    """
    super(TP10X2, self).__setstate__(state)
    if self.makeCells4Ephemeral:
      self.cells4 = Cells4(self.numberOfCols,
                 self.cellsPerColumn,
                 self.activationThreshold,
                 self.minThreshold,
                 self.newSynapseCount,
                 self.segUpdateValidDuration,
                 self.initialPerm,
                 self.connectedPerm,
                 self.permanenceMax,
                 self.permanenceDec,
                 self.permanenceInc,
                 self.globalDecay,
                 self.doPooling,
                 self.seed,
                 self.allocateStatesInCPP,
                 self.checkSynapseConsistency)

      self.cells4.setVerbosity(self.verbosity)
      self.cells4.setPamLength(self.pamLength)
      self.cells4.setMaxAge(self.maxAge)
      self.cells4.setMaxInfBacktrack(self.maxInfBacktrack)
      self.cells4.setMaxLrnBacktrack(self.maxLrnBacktrack)
      self.cells4.setMaxSeqLength(self.maxSeqLength)
      self.cells4.setMaxSegmentsPerCell(self.maxSegmentsPerCell)
      self.cells4.setMaxSynapsesPerCell(self.maxSynapsesPerSegment)

    # Reset internal C++ pointers to states
    self._setStatePointers()


  def _getEphemeralMembers(self):
    """
    List of our member variables that we don't need to be saved
    """
    e = TP._getEphemeralMembers(self)
    if self.makeCells4Ephemeral:
      e.extend(['cells4'])
    return e


  def _initEphemerals(self):
    """
    Initialize all ephemeral members after being restored to a pickled state.
    """
    TP._initEphemerals(self)
    #---------------------------------------------------------------------------------
    # cells4 specific initialization

    # If True, let C++ allocate memory for activeState, predictedState, and
    # learnState. In this case we can retrieve copies of these states but can't
    # set them directly from Python. If False, Python can allocate them as
    # numpy arrays and we can pass pointers to the C++ using setStatePointers
    self.allocateStatesInCPP = False

    # Set this to true for debugging or accessing learning states
    self.retrieveLearningStates = False

    if self.makeCells4Ephemeral:
      self.cells4 = Cells4(self.numberOfCols,
                 self.cellsPerColumn,
                 self.activationThreshold,
                 self.minThreshold,
                 self.newSynapseCount,
                 self.segUpdateValidDuration,
                 self.initialPerm,
                 self.connectedPerm,
                 self.permanenceMax,
                 self.permanenceDec,
                 self.permanenceInc,
                 self.globalDecay,
                 self.doPooling,
                 self.seed,
                 self.allocateStatesInCPP,
                 self.checkSynapseConsistency)

      self.cells4.setVerbosity(self.verbosity)
      self.cells4.setPamLength(self.pamLength)
      self.cells4.setMaxAge(self.maxAge)
      self.cells4.setMaxInfBacktrack(self.maxInfBacktrack)
      self.cells4.setMaxLrnBacktrack(self.maxLrnBacktrack)
      self.cells4.setMaxSeqLength(self.maxSeqLength)
      self.cells4.setMaxSegmentsPerCell(self.maxSegmentsPerCell)
      self.cells4.setMaxSynapsesPerCell(self.maxSynapsesPerSegment)

      self._setStatePointers()

  def saveToFile(self, filePath):
    """
    Save Cells4 state to this file
    """
    self.cells4.saveToFile(filePath)

  def loadFromFile(self, filePath):
    """
    Load Cells4 state from this file
    """
    self.cells4.loadFromFile(filePath)


  def __getattr__(self, name):
    """
    Patch __getattr__ so that we can catch the first access to 'cells' and load.

    This function is only called when we try to access an attribute that doesn't
    exist.  We purposely make sure that "self.cells" doesn't exist after
    unpickling so that we'll hit this, then we can load it on the first access.

    If this is called at any other time, it will raise an AttributeError.
    That's because:
    - If 'name' is "cells", after the first call, self._realCells won't exist
      so we'll get an implicit AttributeError.
    - If 'name' isn't "cells", I'd expect our super wouldn't have __getattr__,
      so we'll raise our own Attribute error.  If the super did get __getattr__,
      we'll just return what it gives us.
    """

    try:
      return super(TP, self).__getattr__(name)
    except AttributeError:
      raise AttributeError("'TP' object has no attribute '%s'" % name)


  def compute(self, bottomUpInput, enableLearn, computeInfOutput=None):
    """ Handle one compute, possibly learning.

    By default, we don't compute the inference output when learning because it
    slows things down, but you can override this by passing in True for
    computeInfOutput
    """
    # The C++ TP takes 32 bit floats as input. uint32 works as well since the
    # code only checks whether elements are non-zero
    assert (bottomUpInput.dtype == numpy.dtype('float32')) or \
           (bottomUpInput.dtype == numpy.dtype('uint32')) or \
           (bottomUpInput.dtype == numpy.dtype('int32'))

    self.iterationIdx = self.iterationIdx + 1

    print "ITER: ", self.iterationIdx

    #if self.iterationIdx >= 1000040:
    #  self.verbosity=4                           # DEBUG
    #  self.cells4.setVerbosity(self.verbosity)   # DEBUG

    # As a speed optimization for now (until we need online learning), skip
    #  computing the inference output while learning
    if computeInfOutput is None:
      if enableLearn:
        computeInfOutput = False
      else:
        computeInfOutput = True

    # ====================================================================
    # Run compute and retrieve selected state and member variables
    self._setStatePointers()
    y = self.cells4.compute(bottomUpInput, computeInfOutput, enableLearn)
    if self.iterationIdx == 48:
      onbits = set(numpy.nonzero(y)[0])
      onbits48 = set([1544,  2400,  2401,  2402,  2403,  2404,  2405,  2406,  2407,
        2408,  2409,  2410,  2411,  2412,  2413,  2414,  2415,  2416,
        2417,  2418,  2419,  2420,  2421,  2422,  2423,  2424,  2425,
        2426,  2427,  2428,  2429,  2430,  2431,  7117,  7420,  9344,
        9345,  9346,  9347,  9348,  9349,  9350,  9351,  9352,  9353,
        9354,  9355,  9356,  9357,  9358,  9359,  9360,  9361,  9362,
        9363,  9364,  9365,  9366,  9367,  9368,  9369,  9370,  9371,
        9372,  9373,  9374,  9375,  9495, 10684, 11851, 17525, 17857,
       20960, 20961, 20962, 20963, 20964, 20965, 20966, 20967, 20968,
       20969, 20970, 20971, 20972, 20973, 20974, 20975, 20976, 20977,
       20978, 20979, 20980, 20981, 20982, 20983, 20984, 20985, 20986,
       20987, 20988, 20989, 20990, 20991, 22064, 22624, 22625, 22626,
       22627, 22628, 22629, 22630, 22631, 22632, 22633, 22634, 22635,
       22636, 22637, 22638, 22639, 22640, 22641, 22642, 22643, 22644,
       22645, 22646, 22647, 22648, 22649, 22650, 22651, 22652, 22653,
       22654, 22655, 23808, 23809, 23810, 23811, 23812, 23813, 23814,
       23815, 23816, 23817, 23818, 23819, 23820, 23821, 23822, 23823,
       23824, 23825, 23826, 23827, 23828, 23829, 23830, 23831, 23832,
       23833, 23834, 23835, 23836, 23837, 23838, 23839, 27040, 27041,
       27042, 27043, 27044, 27045, 27046, 27047, 27048, 27049, 27050,
       27051, 27052, 27053, 27054, 27055, 27056, 27057, 27058, 27059,
       27060, 27061, 27062, 27063, 27064, 27065, 27066, 27067, 27068,
       27069, 27070, 27071, 27136, 27137, 27138, 27139, 27140, 27141,
       27142, 27143, 27144, 27145, 27146, 27147, 27148, 27149, 27150,
       27151, 27152, 27153, 27154, 27155, 27156, 27157, 27158, 27159,
       27160, 27161, 27162, 27163, 27164, 27165, 27166, 27167, 27392,
       27393, 27394, 27395, 27396, 27397, 27398, 27399, 27400, 27401,
       27402, 27403, 27404, 27405, 27406, 27407, 27408, 27409, 27410,
       27411, 27412, 27413, 27414, 27415, 27416, 27417, 27418, 27419,
       27420, 27421, 27422, 27423, 27520, 27521, 27522, 27523, 27524,
       27525, 27526, 27527, 27528, 27529, 27530, 27531, 27532, 27533,
       27534, 27535, 27536, 27537, 27538, 27539, 27540, 27541, 27542,
       27543, 27544, 27545, 27546, 27547, 27548, 27549, 27550, 27551,
       32096, 32097, 32098, 32099, 32100, 32101, 32102, 32103, 32104,
       32105, 32106, 32107, 32108, 32109, 32110, 32111, 32112, 32113,
       32114, 32115, 32116, 32117, 32118, 32119, 32120, 32121, 32122,
       32123, 32124, 32125, 32126, 32127, 32940, 32960, 32961, 32962,
       32963, 32964, 32965, 32966, 32967, 32968, 32969, 32970, 32971,
       32972, 32973, 32974, 32975, 32976, 32977, 32978, 32979, 32980,
       32981, 32982, 32983, 32984, 32985, 32986, 32987, 32988, 32989,
       32990, 32991, 35909, 38055, 39808, 39809, 39810, 39811, 39812,
       39813, 39814, 39815, 39816, 39817, 39818, 39819, 39820, 39821,
       39822, 39823, 39824, 39825, 39826, 39827, 39828, 39829, 39830,
       39831, 39832, 39833, 39834, 39835, 39836, 39837, 39838, 39839,
       40960, 40961, 40962, 40963, 40964, 40965, 40966, 40967, 40968,
       40969, 40970, 40971, 40972, 40973, 40974, 40975, 40976, 40977,
       40978, 40979, 40980, 40981, 40982, 40983, 40984, 40985, 40986,
       40987, 40988, 40989, 40990, 40991, 41092, 49792, 49793, 49794,
       49795, 49796, 49797, 49798, 49799, 49800, 49801, 49802, 49803,
       49804, 49805, 49806, 49807, 49808, 49809, 49810, 49811, 49812,
       49813, 49814, 49815, 49816, 49817, 49818, 49819, 49820, 49821,
       49822, 49823, 53984, 53985, 53986, 53987, 53988, 53989, 53990,
       53991, 53992, 53993, 53994, 53995, 53996, 53997, 53998, 53999,
       54000, 54001, 54002, 54003, 54004, 54005, 54006, 54007, 54008,
       54009, 54010, 54011, 54012, 54013, 54014, 54015, 55520, 55521,
       55522, 55523, 55524, 55525, 55526, 55527, 55528, 55529, 55530,
       55531, 55532, 55533, 55534, 55535, 55536, 55537, 55538, 55539,
       55540, 55541, 55542, 55543, 55544, 55545, 55546, 55547, 55548,
       55549, 55550, 55551, 56544, 56545, 56546, 56547, 56548, 56549,
       56550, 56551, 56552, 56553, 56554, 56555, 56556, 56557, 56558,
       56559, 56560, 56561, 56562, 56563, 56564, 56565, 56566, 56567,
       56568, 56569, 56570, 56571, 56572, 56573, 56574, 56575, 57056,
       57057, 57058, 57059, 57060, 57061, 57062, 57063, 57064, 57065,
       57066, 57067, 57068, 57069, 57070, 57071, 57072, 57073, 57074,
       57075, 57076, 57077, 57078, 57079, 57080, 57081, 57082, 57083,
       57084, 57085, 57086, 57087, 57824, 57825, 57826, 57827, 57828,
       57829, 57830, 57831, 57832, 57833, 57834, 57835, 57836, 57837,
       57838, 57839, 57840, 57841, 57842, 57843, 57844, 57845, 57846,
       57847, 57848, 57849, 57850, 57851, 57852, 57853, 57854, 57855,
       59680, 59681, 59682, 59683, 59684, 59685, 59686, 59687, 59688,
       59689, 59690, 59691, 59692, 59693, 59694, 59695, 59696, 59697,
       59698, 59699, 59700, 59701, 59702, 59703, 59704, 59705, 59706,
       59707, 59708, 59709, 59710, 59711, 60128, 60129, 60130, 60131,
       60132, 60133, 60134, 60135, 60136, 60137, 60138, 60139, 60140,
       60141, 60142, 60143, 60144, 60145, 60146, 60147, 60148, 60149,
       60150, 60151, 60152, 60153, 60154, 60155, 60156, 60157, 60158,
       60159, 60801, 61312, 61313, 61314, 61315, 61316, 61317, 61318,
       61319, 61320, 61321, 61322, 61323, 61324, 61325, 61326, 61327,
       61328, 61329, 61330, 61331, 61332, 61333, 61334, 61335, 61336,
       61337, 61338, 61339, 61340, 61341, 61342, 61343, 62928, 63424,
       63425, 63426, 63427, 63428, 63429, 63430, 63431, 63432, 63433,
       63434, 63435, 63436, 63437, 63438, 63439, 63440, 63441, 63442,
       63443, 63444, 63445, 63446, 63447, 63448, 63449, 63450, 63451,
       63452, 63453, 63454, 63455, 63787, 63808, 63809, 63810, 63811,
       63812, 63813, 63814, 63815, 63816, 63817, 63818, 63819, 63820,
       63821, 63822, 63823, 63824, 63825, 63826, 63827, 63828, 63829,
       63830, 63831, 63832, 63833, 63834, 63835, 63836, 63837, 63838, 63839])

      if not onbits == onbits48:
        print onbits - onbits48
        print onbits48 - onbits
        assert False

    self.currentOutput = y.reshape((self.numberOfCols, self.cellsPerColumn))
    self.avgLearnedSeqLength = self.cells4.getAvgLearnedSeqLength()
    self._copyAllocatedStates()

    # if self.seed == 123455:

    #   f = pkg_resources.resource_filename("tmp_mess", "computedOutput.txt")

    #   # with open(f, 'r') as fin:
    #   #   reader=csv.reader(fin)
    #   #   for i,row in enumerate(reader):
    #   #     if i+1 == self.iterationIdx:
    #   #       assert int(row[0]) == self.iterationIdx, str((row[0], self.iterationIdx))
    #   #       for j, x in enumerate(row[1:]):
    #   #         assert int(x) == y[j], str((i, j, x, y[j]))

    # if self.iterationIdx <= 476:
    #   with open(f, 'a') as output_file:
    #     numpy.savetxt(output_file, numpy.array([[self.iterationIdx]]), newline=',', fmt='%d')
    #     numpy.savetxt(output_file, numpy.atleast_2d(y), delimiter=',', fmt='%d')


    # ========================================================================
    # Update the prediction score stats
    # Learning always includes inference
    if self.collectStats:
      activeColumns = bottomUpInput.nonzero()[0]
      if computeInfOutput:
        predictedState = self.infPredictedState['t-1']
      else:
        predictedState = self.lrnPredictedState['t-1']
      self._updateStatsInferEnd(self._internalStats,
                                activeColumns,
                                predictedState,
                                self.colConfidence['t-1'])



    # Finally return the TP output
    output = self.computeOutput()

    # Print diagnostic information based on the current verbosity level
    self.printComputeEnd(output, learn=enableLearn)

    self.resetCalled = False
    return output

  def inferPhase2(self):
    """
    This calls phase 2 of inference (used in multistep prediction).
    """

    self._setStatePointers()
    self.cells4.inferPhase2()
    self._copyAllocatedStates()


  def getLearnActiveStateT(self):
    if self.verbosity > 1 or self.retrieveLearningStates:
      return self.lrnActiveState['t']
    else:
      (activeT, _, _, _) = self.cells4.getLearnStates()
      return activeT.reshape((self.numberOfCols, self.cellsPerColumn))


  def _copyAllocatedStates(self):
    """If state is allocated in CPP, copy over the data into our numpy arrays."""

    # Get learn states if we need to print them out
    if self.verbosity > 1 or self.retrieveLearningStates:
      (activeT, activeT1, predT, predT1) = self.cells4.getLearnStates()
      self.lrnActiveState['t-1'] = activeT1.reshape((self.numberOfCols, self.cellsPerColumn))
      self.lrnActiveState['t'] = activeT.reshape((self.numberOfCols, self.cellsPerColumn))
      self.lrnPredictedState['t-1'] = predT1.reshape((self.numberOfCols, self.cellsPerColumn))
      self.lrnPredictedState['t'] = predT.reshape((self.numberOfCols, self.cellsPerColumn))

    if self.allocateStatesInCPP:
      assert False
      (activeT, activeT1, predT, predT1, colConfidenceT, colConfidenceT1, confidenceT,
       confidenceT1) = self.cells4.getStates()
      self.confidence['t-1'] = confidenceT1.reshape((self.numberOfCols, self.cellsPerColumn))
      self.confidence['t'] = confidenceT.reshape((self.numberOfCols, self.cellsPerColumn))
      self.colConfidence['t'] = colConfidenceT.reshape(self.numberOfCols)
      self.colConfidence['t-1'] = colConfidenceT1.reshape(self.numberOfCols)
      self.infActiveState['t-1'] = activeT1.reshape((self.numberOfCols, self.cellsPerColumn))
      self.infActiveState['t'] = activeT.reshape((self.numberOfCols, self.cellsPerColumn))
      self.infPredictedState['t-1'] = predT1.reshape((self.numberOfCols, self.cellsPerColumn))
      self.infPredictedState['t'] = predT.reshape((self.numberOfCols, self.cellsPerColumn))

  def _setStatePointers(self):
    """If we are having CPP use numpy-allocated buffers, set these buffer
    pointers. This is a relatively fast operation and, for safety, should be
    done before every call to the cells4 compute methods.  This protects us
    in situations where code can cause Python or numpy to create copies."""
    if not self.allocateStatesInCPP:
      self.cells4.setStatePointers(
          self.infActiveState["t"], self.infActiveState["t-1"],
          self.infPredictedState["t"], self.infPredictedState["t-1"],
          self.colConfidence["t"], self.colConfidence["t-1"],
          self.cellConfidence["t"], self.cellConfidence["t-1"])


  def reset(self):
    """ Reset the state of all cells.
    This is normally used between sequences while training. All internal states
    are reset to 0.
    """
    if self.verbosity >= 3:
      print "TP Reset"
    self._setStatePointers()
    self.cells4.reset()
    TP.reset(self)


  def finishLearning(self):
    """Called when learning has been completed. This method just calls
    trimSegments. (finishLearning is here for backward compatibility)
    """
    # Keep weakly formed synapses around because they contain confidence scores
    #  for paths out of learned sequenced and produce a better prediction than
    #  chance.
    self.trimSegments(minPermanence=0.0001)


  def trimSegments(self, minPermanence=None, minNumSyns=None):
    """This method deletes all synapses where permanence value is strictly
    less than self.connectedPerm. It also deletes all segments where the
    number of connected synapses is strictly less than self.activationThreshold.
    Returns the number of segments and synapses removed. This often done
    after formal learning has completed so that subsequence inference runs
    faster.

    Parameters:
    --------------------------------------------------------------
    minPermanence:      Any syn whose permamence is 0 or < minPermanence will
                        be deleted. If None is passed in, then
                        self.connectedPerm is used.
    minNumSyns:         Any segment with less than minNumSyns synapses remaining
                        in it will be deleted. If None is passed in, then
                        self.activationThreshold is used.
    retval:             (numSegsRemoved, numSynsRemoved)
    """

    # Fill in defaults
    if minPermanence is None:
      minPermanence = 0.0
    if minNumSyns is None:
      minNumSyns = 0

    # Print all cells if verbosity says to
    if self.verbosity >= 5:
      print "Cells, all segments:"
      self.printCells(predictedOnly=False)

    return self.cells4.trimSegments(minPermanence=minPermanence, minNumSyns=minNumSyns)

  ################################################################################
  # The following print functions for debugging.
  ################################################################################


  def printSegment(self, s):

    # TODO: need to add C++ accessors to get segment details
    assert False

    prevAct = self.getSegmentActivityLevel(s, 't-1')
    currAct = self.getSegmentActivityLevel(s, 't')

    # Sequence segment or pooling segment
    if s[0][1] == True:
      print "S",
    else:
      print 'P',

    # Frequency count
    print s[0][2],

    if self.isSegmentActive(s, 't'):
      ss = '[' + str(currAct) + ']'
    else:
      ss = str(currAct)
    ss = ss + '/'
    if self.isSegmentActive(s,'t-1'):
      ss = ss + '[' + str(prevAct) + ']'
    else:
      ss = ss + str(prevAct)
    ss = ss + ':'
    print ss,

    for i,synapse in enumerate(s[1:]):

      if synapse[2] >= self.connectedPerm:
        ss = '['
      else:
        ss = ''
      ss = ss + str(synapse[0]) + '/' + str(synapse[1])
      if self.infActiveState['t'][synapse[0],synapse[1]] == 1:
        ss = ss + '/ON'
      ss = ss + '/'
      sf = str(synapse[2])
      ss = ss + sf[:4]
      if synapse[2] >= self.connectedPerm:
        ss = ss + ']'
      if i < len(s)-2:
        ss = ss + ' |'
      print ss,

    if self.verbosity > 3:
      if self.isSegmentActive(s, 't') and \
             prevAct < self.activationThreshold and currAct >= self.activationThreshold:
        print "reached activation",
      if prevAct < self.minThreshold and currAct >= self.minThreshold:
        print "reached min threshold",
      if self.isSegmentActive(s, 't-1') and \
             prevAct >= self.activationThreshold and currAct < self.activationThreshold:
        print "dropped below activation",
      if prevAct >= self.minThreshold and currAct < self.minThreshold:
        print "dropped below min",
      if self.isSegmentActive(s, 't') and self.isSegmentActive(s, 't-1') and \
             prevAct >= self.activationThreshold and currAct >= self.activationThreshold:
        print "maintained activation",

  def printSegmentUpdates(self):
    # TODO: need to add C++ accessors to implement this method
    assert False
    print "=== SEGMENT UPDATES ===, Num = ",len(self.segmentUpdates)
    for key, updateList in self.segmentUpdates.iteritems():
      c,i = key[0],key[1]
      print c,i,updateList


  def slowIsSegmentActive(self, seg, timeStep):
    """
    A segment is active if it has >= activationThreshold connected
    synapses that are active due to infActiveState.

    """

    numSyn = seg.size()
    numActiveSyns = 0
    for synIdx in xrange(numSyn):
      if seg.getPermanence(synIdx) < self.connectedPerm:
        continue
      sc, si = self.getColCellIdx(seg.getSrcCellIdx(synIdx))
      if self.infActiveState[timeStep][sc, si]:
        numActiveSyns += 1
        if numActiveSyns >= self.activationThreshold:
          return True

    return numActiveSyns >= self.activationThreshold


  def printCell(self, c, i, onlyActiveSegments=False):

    nSegs = self.cells4.nSegmentsOnCell(c,i)
    if nSegs > 0:
      segList = self.cells4.getNonEmptySegList(c,i)
      gidx = c * self.cellsPerColumn + i
      print "Column", c, "Cell", i, "(%d)"%(gidx),":", nSegs, "segment(s)"
      for k,segIdx in enumerate(segList):
        seg = self.cells4.getSegment(c, i, segIdx)
        isActive = self.slowIsSegmentActive(seg, 't')
        if onlyActiveSegments and not isActive:
          continue
        isActiveStr = "*" if isActive else " "
        print "  %sSeg #%-3d" % (isActiveStr, segIdx),
        print seg.size(),
        print seg.isSequenceSegment(), "%9.7f" % (seg.dutyCycle(
              self.cells4.getNLrnIterations(), False, True)),

        # numPositive/totalActivations
        print "(%4d/%-4d)" % (seg.getPositiveActivations(),
                           seg.getTotalActivations()),
        # Age
        print "%4d" % (self.cells4.getNLrnIterations()
                       - seg.getLastActiveIteration()),

        numSyn = seg.size()
        for s in xrange(numSyn):
          sc, si = self.getColCellIdx(seg.getSrcCellIdx(s))
          print "[%d,%d]%4.2f"%(sc, si, seg.getPermanence(s)),
        print


  def getAvgLearnedSeqLength(self):
    """ Return our moving average of learned sequence length.
    """
    return self.cells4.getAvgLearnedSeqLength()


  def getColCellIdx(self, idx):
    """Get column and cell within column from a global cell index.
    The global index is idx = colIdx * nCellsPerCol() + cellIdxInCol
    This method returns (colIdx, cellIdxInCol)
    """
    c = idx//self.cellsPerColumn
    i = idx - c*self.cellsPerColumn
    return c,i


  def getSegmentOnCell(self, c, i, segIdx):
    """Return segment number segIdx on cell (c,i).
    Returns the segment as following list:
      [  [segIdx, sequenceSegmentFlag, positive activations,
          total activations, last active iteration],
         [col1, idx1, perm1],
         [col2, idx2, perm2], ...
      ]

    """
    segList = self.cells4.getNonEmptySegList(c,i)
    seg = self.cells4.getSegment(c, i, segList[segIdx])
    numSyn = seg.size()
    assert numSyn != 0

    # Accumulate segment information
    result = []
    result.append([int(segIdx), bool(seg.isSequenceSegment()),
                   seg.getPositiveActivations(),
                   seg.getTotalActivations(), seg.getLastActiveIteration(),
                   seg.getLastPosDutyCycle(),
                   seg.getLastPosDutyCycleIteration()])

    for s in xrange(numSyn):
      sc, si = self.getColCellIdx(seg.getSrcCellIdx(s))
      result.append([int(sc), int(si), seg.getPermanence(s)])

    return result


  def getNumSegments(self):
    """ Return the total number of segments. """
    return self.cells4.nSegments()


  def getNumSynapses(self):
    """ Return the total number of synapses. """
    return self.cells4.nSynapses()


  def getNumSegmentsInCell(self, c, i):
    """ Return the total number of segments in cell (c,i)"""
    return self.cells4.nSegmentsOnCell(c,i)


  def getSegmentInfo(self, collectActiveData = False):
    """Returns information about the distribution of segments, synapses and
    permanence values in the current TP. If requested, also returns information
    regarding the number of currently active segments and synapses.

    The method returns the following tuple:

    (
      nSegments,        # total number of segments
      nSynapses,        # total number of synapses
      nActiveSegs,      # total no. of active segments
      nActiveSynapses,  # total no. of active synapses
      distSegSizes,     # a dict where d[n] = number of segments with n synapses
      distNSegsPerCell, # a dict where d[n] = number of cells with n segments
      distPermValues,   # a dict where d[p] = number of synapses with perm = p/10
      distAges,         # a list of tuples (ageRange, numSegments)
    )

    nActiveSegs and nActiveSynapses are 0 if collectActiveData is False
    """

    # Requires appropriate accessors in C++ cells4 (currently unimplemented)
    assert collectActiveData == False

    nSegments, nSynapses = self.getNumSegments(), self.cells4.nSynapses()
    distSegSizes, distNSegsPerCell = {}, {}
    nActiveSegs, nActiveSynapses = 0, 0
    distPermValues = {}   # Num synapses with given permanence values

    numAgeBuckets = 20
    distAges = []
    ageBucketSize = int((self.iterationIdx+20) / 20)
    for i in range(numAgeBuckets):
      distAges.append(['%d-%d' % (i*ageBucketSize, (i+1)*ageBucketSize-1), 0])


    for c in xrange(self.numberOfCols):
      for i in xrange(self.cellsPerColumn):

        # Update histogram counting cell sizes
        nSegmentsThisCell = self.getNumSegmentsInCell(c,i)
        if nSegmentsThisCell > 0:
          if distNSegsPerCell.has_key(nSegmentsThisCell):
            distNSegsPerCell[nSegmentsThisCell] += 1
          else:
            distNSegsPerCell[nSegmentsThisCell] = 1

          # Update histogram counting segment sizes.
          segList = self.cells4.getNonEmptySegList(c,i)
          for segIdx in xrange(nSegmentsThisCell):
            seg = self.getSegmentOnCell(c, i, segIdx)
            nSynapsesThisSeg = len(seg) - 1
            if nSynapsesThisSeg > 0:
              if distSegSizes.has_key(nSynapsesThisSeg):
                distSegSizes[nSynapsesThisSeg] += 1
              else:
                distSegSizes[nSynapsesThisSeg] = 1

              # Accumulate permanence value histogram
              for syn in seg[1:]:
                p = int(syn[2]*10)
                if distPermValues.has_key(p):
                  distPermValues[p] += 1
                else:
                  distPermValues[p] = 1

            segObj = self.cells4.getSegment(c, i, segList[segIdx])
            age = self.iterationIdx - segObj.getLastActiveIteration()
            ageBucket = int(age/ageBucketSize)
            distAges[ageBucket][1] += 1


    return (nSegments, nSynapses, nActiveSegs, nActiveSynapses, \
            distSegSizes, distNSegsPerCell, distPermValues, distAges)


  def getActiveSegment(self, c,i, timeStep):
    """ For a given cell, return the segment with the strongest _connected_
    activation, i.e. sum up the activations of the connected synapses of the
    segments only. That is, a segment is active only if it has enough connected
    synapses.
    """

    # TODO: add C++ accessor to implement this
    assert False


  def getBestMatchingCell(self, c, timeStep, learnState = False):
    """Find weakly activated cell in column. Returns index and segment of most
    activated segment above minThreshold.
    """

    # TODO: add C++ accessor to implement this
    assert False


  def getLeastAllocatedCell(self, c):
    """For the given column, return the cell with the fewest number of
    segments."""

    # TODO: add C++ accessor to implement this or implement our own variation
    assert False

  ################################################################################
  # The following methods are implemented in the base class but should never
  # be called in this implementation.
  ################################################################################


  def isSegmentActive(self, seg, timeStep):
    """    """
    # Should never be called in this subclass
    assert False


  def getSegmentActivityLevel(self, seg, timeStep, connectedSynapsesOnly =False,
                              learnState = False):
    """   """
    # Should never be called in this subclass
    assert False


  def isSequenceSegment(self, s):
    """   """
    # Should never be called in this subclass
    assert False


  def getBestMatchingSegment(self, c, i, timeStep, learnState = False):
    """     """
    # Should never be called in this subclass
    assert False


  def getSegmentActiveSynapses(self, c,i,s, timeStep, newSynapses =False):
    """  """
    # Should never be called in this subclass
    assert False


  def updateSynapse(self, segment, synapse, delta):
    """ """
    # Should never be called in this subclass
    assert False


  def adaptSegment(self, update, positiveReinforcement):
    """    """
    # Should never be called in this subclass
    assert False
