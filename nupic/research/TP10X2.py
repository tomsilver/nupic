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
    print "Bottom up input: ", set(numpy.nonzero(bottomUpInput)[0])

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

    if self.iterationIdx <= 37:
      for i in range(self.cellsPerColumn):
        self.printCell(c=1280, i=i)

    y = self.cells4.compute(bottomUpInput, computeInfOutput, enableLearn)

    # if self.iterationIdx == 2:
    #   print "QV: "
    #   onColumns = set(numpy.nonzero(bottomUpInput)[0])
    #   onColumns2 = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 277, 278, 755, 260, 265, 768, 261, 266, 262, 263, 769, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 267, 268, 258, 269, 270, 761, 771, 770, 271, 272, 766, 273, 259, 274, 751, 752, 753, 754, 275, 756, 757, 758, 759, 760, 276, 762, 763, 764, 765, 264, 767])

    #   if not onColumns == onColumns2:
    #     print "Columns are different"
    #     print onColumns - onColumns2
    #     print onColumns2 - onColumns
    #     assert False

    # if self.iterationIdx == 2:
    #   print "QT: "
    #   onbits = set(numpy.nonzero(y)[0])
    #   onbits2 = set([40960, 40961, 40962, 7851, 40964, 40965, 40966, 40967, 40968, 7852, 40970, 40971, 40972, 40973, 40974, 7853, 40976, 40977, 40978, 40963, 40980, 5806, 40982, 40983, 40984, 40985, 40986, 7855, 39827, 39826, 40990, 40991, 5808, 61787, 58607, 5809, 57081, 5810, 61789, 40987, 7859, 40969, 7860, 39828, 61791, 7861, 6208, 6209, 6210, 6211, 6212, 6213, 6214, 6215, 6216, 6217, 6218, 6219, 6220, 6221, 6222, 6223, 6224, 6225, 6226, 6227, 6228, 6229, 6230, 6231, 6232, 6233, 6234, 6235, 6236, 6237, 6238, 6239, 5819, 57083, 5820, 40989, 48885, 5821, 40979, 5822, 57080, 5823, 40981, 47232, 47233, 47234, 47235, 47236, 47237, 47238, 47239, 47240, 47241, 47242, 47243, 47244, 7847, 47246, 47247, 47248, 47249, 47250, 47251, 47252, 47253, 47254, 47255, 47256, 47257, 47258, 47259, 47260, 47261, 47262, 47263, 24736, 24737, 24738, 24739, 24740, 24741, 24742, 24743, 24744, 24745, 24746, 24747, 24748, 24749, 24750, 24751, 24752, 24753, 24754, 24755, 24756, 24757, 24758, 24759, 24760, 24761, 24762, 24763, 24764, 24765, 24766, 24767, 57086, 55995, 57074, 55058, 7848, 57087, 55520, 55521, 55522, 55523, 55524, 55525, 55526, 55527, 55528, 55529, 55530, 55531, 55532, 55533, 55534, 55535, 55536, 55537, 55538, 55539, 55540, 55541, 55542, 55543, 55544, 55545, 55546, 55547, 55548, 55549, 55550, 55551, 8448, 8449, 8450, 8451, 8452, 8453, 8454, 8455, 8456, 8457, 8458, 8459, 8460, 8461, 8462, 8463, 8464, 8465, 8466, 8467, 8468, 8469, 8470, 8471, 8472, 8473, 8474, 8475, 8476, 8477, 8478, 8479, 59680, 59681, 59682, 59683, 55061, 59685, 59686, 59687, 59688, 59689, 39836, 59691, 59692, 59693, 59694, 59695, 59696, 59697, 55049, 59699, 59700, 59701, 59702, 59703, 59704, 59705, 59706, 59707, 59708, 59709, 59710, 59711, 61760, 32480, 61762, 61763, 57076, 61765, 61766, 32481, 39837, 55282, 61770, 61771, 61772, 32482, 61774, 61775, 61776, 61777, 61778, 32483, 61780, 61781, 61782, 61783, 61784, 32484, 61786, 57066, 61788, 39829, 61790, 32485, 57070, 52795, 32486, 39838, 32487, 32488, 55046, 32489, 55295, 48871, 32490, 56561, 32491, 39839, 60587, 32492, 32493, 55047, 32494, 57058, 32495, 48874, 32496, 57082, 32497, 39815, 32498, 55048, 40988, 32499, 32500, 32501, 16832, 16833, 16834, 16835, 16836, 16837, 16838, 16839, 16840, 16841, 16842, 16843, 16844, 16845, 16846, 16847, 16848, 16849, 16850, 16851, 16852, 16853, 16854, 16855, 16856, 16857, 16858, 16859, 16860, 16861, 16862, 16863, 32507, 58610, 32508, 55050, 52789, 32509, 32510, 32505, 32511, 55064, 56551, 27392, 58597, 55041, 55042, 57079, 48872, 55043, 55044, 55045, 16928, 16929, 16930, 16931, 16932, 16933, 16934, 16935, 16936, 16937, 16938, 16939, 16940, 5815, 16942, 16943, 16944, 16945, 16946, 16947, 16948, 16949, 16950, 16951, 16952, 16953, 16954, 16955, 16956, 16957, 16958, 16959, 55051, 55053, 55052, 5816, 7862, 55054, 55055, 58600, 43616, 43617, 43618, 43619, 43620, 43621, 43622, 43623, 43624, 43625, 43626, 43627, 43628, 43629, 43630, 43631, 43632, 43633, 43634, 43635, 43636, 43637, 43638, 43639, 43640, 43641, 43642, 43643, 43644, 43645, 43646, 43647, 55062, 55063, 48883, 55065, 58602, 55219, 55066, 55056, 55968, 55969, 55970, 55067, 55972, 55973, 55974, 55975, 55976, 7865, 55978, 55979, 55980, 55981, 55982, 55069, 55984, 55985, 55986, 55987, 55988, 55070, 55990, 55991, 55992, 55993, 55994, 55071, 55057, 55997, 55989, 55999, 31424, 31425, 31426, 31427, 31428, 31429, 31430, 7866, 31432, 31433, 31434, 31435, 31436, 31437, 31438, 31439, 31440, 31441, 31442, 31443, 31444, 31445, 31446, 31447, 31448, 31449, 31450, 31451, 31452, 31453, 31454, 31455, 26621, 39835, 55059, 57057, 7846, 27393, 27394, 27395, 27396, 27397, 27398, 27399, 27400, 27401, 27402, 27403, 27404, 27405, 27406, 27407, 27408, 27409, 27410, 27411, 27412, 27413, 27414, 27415, 27416, 27417, 27418, 27419, 27420, 27421, 27422, 27423, 35616, 35617, 35618, 35619, 35620, 35621, 35622, 35623, 35624, 35625, 35626, 35627, 35628, 35629, 35630, 35631, 35632, 35633, 35634, 35635, 35636, 35637, 35638, 35639, 35640, 35641, 35642, 35643, 35644, 35645, 35646, 35647, 11072, 11073, 11074, 11075, 11076, 11077, 11078, 11079, 11080, 11081, 11082, 11083, 11084, 11085, 11086, 11087, 11088, 11089, 11090, 11091, 11092, 11093, 11094, 11095, 11096, 11097, 11098, 11099, 11100, 11101, 11102, 11103, 59690, 57085, 58609, 39818, 48888, 58599, 55268, 57064, 29568, 29569, 29570, 29571, 29572, 29573, 29574, 29575, 29576, 29577, 29578, 29579, 29580, 29581, 29582, 29583, 29584, 29585, 29586, 29587, 29588, 29589, 29590, 29591, 29592, 29593, 29594, 29595, 29596, 29597, 29598, 29599, 55291, 58608, 60585, 60589, 55270, 58612, 60577, 57072, 55271, 55285, 58613, 60582, 55040, 55231, 55272, 40951, 27648, 27649, 27650, 27651, 27652, 27653, 27654, 27655, 27656, 27657, 27658, 27659, 27660, 27661, 27662, 27663, 27664, 27665, 27666, 27667, 27668, 27669, 27670, 27671, 27672, 27673, 27674, 27675, 27676, 27677, 27678, 27679, 57059, 55274, 55283, 48864, 60597, 55266, 57071, 57056, 55275, 55068, 48865, 57073, 57068, 55086, 48876, 55276, 40955, 58611, 48866, 61761, 57078, 55277, 40956, 55273, 48867, 60576, 58601, 60578, 60579, 60580, 60581, 55996, 60583, 60584, 55278, 60586, 40957, 60588, 58622, 60590, 60591, 60592, 60593, 60594, 60595, 60596, 48868, 60598, 60599, 60600, 60601, 60602, 60603, 60604, 60605, 60606, 60607, 57060, 55279, 40958, 32502, 48869, 56544, 56545, 56546, 56547, 56548, 55280, 56550, 40959, 56552, 56553, 56554, 56555, 56556, 56557, 56558, 32503, 56560, 48870, 56562, 56563, 56564, 56565, 56566, 56567, 56568, 56569, 56570, 56571, 56572, 56573, 56574, 56575, 39808, 55281, 61785, 39809, 59698, 32504, 16941, 39811, 57075, 39812, 39813, 11552, 11553, 11554, 11555, 11556, 11557, 11558, 11559, 11560, 11561, 11562, 11563, 11564, 11565, 11566, 11567, 11568, 11569, 11570, 11571, 11572, 11573, 11574, 11575, 11576, 11577, 11578, 11579, 11580, 11581, 11582, 11583, 58592, 39819, 58593, 32506, 48873, 58594, 39821, 58595, 39822, 55267, 58596, 39823, 55284, 56549, 32096, 32097, 32098, 32099, 32100, 32101, 32102, 32103, 32104, 32105, 32106, 32107, 32108, 32109, 32110, 32111, 32112, 32113, 32114, 32115, 32116, 32117, 32118, 32119, 32120, 32121, 32122, 32123, 32124, 32125, 32126, 32127, 58603, 39830, 48875, 58604, 57061, 39831, 58605, 39832, 58606, 39833, 55286, 56559, 39834, 13728, 13729, 13730, 13731, 13732, 13733, 13734, 13735, 13736, 13737, 13738, 13739, 13740, 13741, 13742, 13743, 13744, 13745, 13746, 13747, 13748, 13749, 13750, 13751, 13752, 13753, 13754, 13755, 13756, 13757, 13758, 13759, 48877, 58614, 55201, 55083, 58615, 58616, 55998, 55203, 55288, 58617, 55204, 58618, 48878, 61767, 58619, 58620, 55207, 58621, 57062, 55289, 55971, 48894, 58623, 48879, 61768, 55213, 55290, 57084, 48880, 39810, 52768, 52769, 52770, 52771, 52772, 52773, 52774, 52775, 52776, 52777, 52778, 52779, 52780, 52781, 52782, 52783, 52784, 52785, 52786, 52787, 52788, 39817, 52790, 52791, 52792, 52793, 52794, 48881, 52796, 52797, 52798, 52799, 48887, 57065, 55292, 47245, 55225, 48882, 55287, 40544, 40545, 40546, 40547, 40548, 40549, 40550, 40551, 40552, 40553, 40554, 40555, 40556, 40557, 40558, 40559, 40560, 40561, 40562, 40563, 40564, 40565, 40566, 40567, 40568, 40569, 40570, 40571, 40572, 40573, 40574, 40575, 57063, 57067, 55294, 48884, 39814, 5792, 7841, 5794, 7843, 7844, 7845, 5798, 5799, 5800, 5801, 5802, 5803, 5804, 5805, 7854, 5807, 7856, 7857, 7858, 5811, 5812, 5813, 5814, 7863, 7864, 5817, 5818, 7867, 7868, 7869, 7870, 7871, 57069, 40975, 48886, 39816, 61769, 59684, 30432, 30433, 30434, 30435, 30436, 30437, 30438, 30439, 30440, 30441, 30442, 30443, 30444, 30445, 30446, 30447, 30448, 30449, 30450, 30451, 30452, 30453, 30454, 30455, 30456, 30457, 30458, 30459, 30460, 30461, 30462, 30463, 36608, 36609, 36610, 36611, 36612, 36613, 36614, 36615, 36616, 36617, 36618, 36619, 36620, 36621, 36622, 36623, 36624, 36625, 36626, 36627, 36628, 36629, 36630, 36631, 36632, 36633, 36634, 36635, 36636, 36637, 36638, 36639, 55072, 55073, 55074, 55075, 55076, 55077, 55078, 55079, 55080, 55081, 55082, 48889, 55084, 55085, 31431, 55087, 55088, 55089, 55090, 55091, 55092, 55093, 55094, 55095, 55096, 55097, 55098, 55099, 55100, 55101, 55102, 55103, 40928, 55269, 40929, 48890, 39820, 40930, 55264, 40931, 40932, 55983, 40933, 40934, 48891, 40935, 61779, 40936, 40937, 55265, 40938, 40939, 48892, 39825, 40940, 48895, 40941, 55293, 57077, 40942, 55977, 61764, 40943, 55200, 40944, 55202, 48893, 55060, 55205, 55206, 40945, 55208, 55209, 55210, 55211, 55212, 40946, 55214, 55215, 55216, 55217, 55218, 40947, 55220, 55221, 55222, 55223, 55224, 40948, 55226, 55227, 55228, 55229, 55230, 40949, 7840, 39824, 40950, 5793, 26615, 7842, 61773, 40952, 58598, 5795, 40953, 5796, 40954, 5797, 26592, 26593, 26594, 26595, 26596, 26597, 26598, 26599, 26600, 26601, 26602, 26603, 26604, 26605, 26606, 26607, 26608, 26609, 26610, 26611, 26612, 26613, 26614, 7849, 26616, 26617, 26618, 26619, 26620, 7850, 26622, 26623])


    #   if not onbits == onbits2:
    #     print onbits-onbits2
    #     print onbits2-onbits
    #     assert False


      # onbits = set(numpy.nonzero(y)[0])
      # onbits36 = set([40960, 40961, 40962, 40619, 40964, 40965, 40966, 40967, 40968, 40620, 40970, 40971, 40972, 40973, 40974, 40621, 40976, 40977, 40978, 40963, 40980, 40622, 40982, 40983, 40984, 40985, 40986, 40623, 40988, 40989, 40990, 40991, 40624, 40625, 40626, 40987, 40627, 40969, 40628, 40629, 61335, 8544, 40630, 2401, 40631, 2402, 40632, 8547, 40633, 8548, 62924, 40975, 40634, 2405, 40635, 62931, 2406, 40636, 8551, 40637, 2408, 40979, 40638, 2409, 40639, 8554, 40981, 41088, 41089, 41090, 8555, 41092, 41093, 41094, 41095, 41096, 8556, 41098, 41099, 41100, 41101, 41102, 2413, 41104, 41105, 41106, 41107, 41108, 8558, 41110, 41111, 41112, 41113, 41114, 2415, 41116, 41117, 41118, 41119, 32928, 2416, 32930, 32931, 32932, 32933, 32934, 2417, 32936, 32937, 32938, 32939, 32940, 2418, 32942, 32943, 32944, 32945, 32946, 2419, 32948, 32949, 32950, 32951, 32952, 2420, 32954, 32955, 32956, 32957, 32958, 2421, 2422, 62918, 2423, 8568, 2425, 54359, 52457, 61315, 2426, 55520, 55521, 55522, 2427, 55524, 55525, 55526, 55527, 55528, 8572, 55530, 55531, 55532, 55533, 55534, 8573, 55536, 55537, 55538, 55539, 55540, 2430, 55542, 55543, 55544, 55545, 55546, 8575, 55548, 55549, 55550, 55551, 30976, 30977, 30978, 30979, 30980, 30981, 30982, 30983, 30984, 30985, 30986, 30987, 30988, 30989, 30990, 30991, 30992, 30993, 30994, 30995, 30996, 30997, 30998, 30999, 31000, 31001, 31002, 31003, 31004, 31005, 31006, 31007, 63776, 63777, 63778, 60600, 63780, 63781, 63782, 63783, 63784, 63785, 63786, 63787, 62928, 63789, 63790, 63791, 17510, 63793, 63794, 63795, 63796, 63797, 61318, 52617, 63800, 63801, 63802, 63803, 63804, 63805, 63806, 63807, 61319, 52623, 62919, 2400, 8545, 8546, 2403, 2404, 8549, 8550, 2407, 8552, 8553, 2410, 2411, 2412, 8557, 2414, 8559, 8560, 8561, 8562, 8563, 8564, 8565, 8566, 8567, 2424, 8569, 8570, 8571, 2428, 2429, 8574, 2431, 54336, 54337, 35906, 61321, 54339, 61316, 54340, 52456, 54341, 10656, 10657, 10658, 10659, 10660, 10661, 10662, 10663, 10664, 10665, 10666, 10667, 10668, 10669, 10670, 10671, 10672, 10673, 10674, 10675, 10676, 10677, 10678, 10679, 10680, 10681, 10682, 10683, 10684, 10685, 10686, 10687, 27040, 54347, 55541, 27041, 54348, 54367, 27042, 45039, 27043, 54350, 27044, 54351, 27045, 54352, 27046, 54353, 61324, 27047, 54354, 27048, 54355, 27049, 54356, 27050, 54357, 61338, 27051, 54358, 27052, 35927, 27053, 54360, 27054, 54361, 27055, 54362, 27056, 54363, 61326, 27057, 54364, 60576, 27058, 54365, 27059, 54366, 27060, 52473, 27061, 17504, 27062, 17505, 60577, 27063, 17506, 27064, 17507, 27065, 17508, 27066, 17509, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 61329, 17856, 17515, 60579, 17857, 17516, 17858, 17517, 17859, 17518, 17860, 52783, 17519, 61330, 17861, 17520, 60580, 17862, 17521, 17863, 17522, 17864, 17523, 17865, 17524, 61331, 17866, 55529, 17525, 62934, 60581, 17867, 17526, 61312, 17868, 17527, 17869, 17528, 47907, 17870, 17529, 61332, 17871, 17530, 17872, 17531, 17873, 17532, 55079, 17874, 17533, 17875, 17534, 47913, 61333, 17876, 17535, 60583, 17877, 17878, 17879, 62936, 45035, 17880, 41091, 61334, 17881, 47919, 60601, 17882, 62922, 47904, 47905, 47906, 17883, 47908, 47909, 47910, 47911, 47912, 17884, 47914, 47915, 47916, 45036, 47918, 17885, 47920, 47921, 47922, 47923, 47924, 17886, 47926, 41097, 47928, 47929, 47930, 17887, 47932, 47933, 47934, 47925, 52470, 55094, 55075, 45037, 62926, 61336, 40618, 41103, 61313, 52471, 47931, 45038, 55101, 61337, 47935, 52472, 41109, 27520, 27521, 27522, 27523, 27524, 27525, 27526, 27527, 27528, 27529, 27530, 27531, 27532, 27533, 27534, 27535, 27536, 27537, 27538, 27539, 27540, 27541, 27542, 27543, 27544, 27545, 27546, 27547, 27548, 27549, 27550, 27551, 41115, 45040, 60589, 62923, 52474, 7104, 7105, 7106, 7107, 7108, 7109, 7110, 7111, 7112, 7113, 7114, 7115, 7116, 7117, 7118, 7119, 7120, 7121, 7122, 7123, 7124, 7125, 7126, 7127, 7128, 7129, 7130, 7131, 7132, 7133, 7134, 7135, 60582, 61341, 32935, 60584, 61314, 60585, 60586, 60587, 60588, 32941, 60605, 52799, 60590, 62929, 60591, 60592, 63798, 55523, 61328, 60593, 55095, 60594, 32947, 62912, 45051, 55547, 60596, 60578, 35904, 35905, 32959, 35907, 35908, 35909, 35910, 35911, 35912, 35913, 35914, 35915, 35916, 35917, 35918, 35919, 35920, 35921, 35922, 35923, 35924, 35925, 35926, 32953, 35928, 35929, 35930, 35931, 35932, 35933, 35934, 35935, 5216, 5217, 5218, 5219, 5220, 5221, 5222, 5223, 5224, 5225, 5226, 5227, 5228, 5229, 5230, 5231, 5232, 5233, 5234, 5235, 5236, 5237, 5238, 5239, 5240, 5241, 5242, 5243, 5244, 5245, 5246, 5247, 9344, 9345, 9346, 9347, 9348, 9349, 9350, 9351, 9352, 9353, 9354, 9355, 9356, 9357, 9358, 9359, 9360, 9361, 9362, 9363, 9364, 9365, 9366, 9367, 9368, 9369, 9370, 9371, 9372, 9373, 9374, 9375, 38048, 38049, 38050, 38051, 38052, 38053, 38054, 38055, 38056, 38057, 38058, 38059, 38060, 38061, 38062, 38063, 38064, 38065, 38066, 38067, 38068, 38069, 38070, 38071, 38072, 38073, 38074, 38075, 38076, 38077, 38078, 38079, 52768, 52769, 40617, 62917, 52780, 40615, 52770, 52771, 52772, 60599, 52773, 7392, 7393, 7394, 7395, 7396, 7397, 7398, 7399, 7400, 7401, 7402, 7403, 7404, 7405, 7406, 7407, 7408, 7409, 7410, 7411, 7412, 7413, 7414, 7415, 7416, 7417, 7418, 7419, 7420, 7421, 7422, 7423, 9472, 9473, 9474, 9475, 9476, 9477, 9478, 9479, 9480, 9481, 9482, 9483, 9484, 9485, 9486, 9487, 9488, 9489, 9490, 9491, 9492, 9493, 9494, 9495, 9496, 9497, 9498, 9499, 9500, 9501, 9502, 9503, 52784, 62920, 47917, 52785, 45053, 52786, 61320, 52787, 60602, 52788, 62913, 60607, 52789, 52448, 62921, 61323, 52790, 52449, 45054, 52791, 52450, 61325, 52792, 52451, 60603, 52793, 52452, 61327, 52794, 52453, 38240, 38241, 38242, 22075, 38244, 38245, 38246, 38247, 38248, 38249, 38250, 38251, 38252, 38253, 38254, 38255, 38256, 38257, 38258, 38259, 38260, 38261, 38262, 38263, 38264, 38265, 38266, 38267, 38268, 38269, 38270, 38271, 52608, 52609, 52610, 52459, 52612, 52613, 52614, 52615, 52616, 52460, 52618, 52619, 52620, 52621, 52622, 52461, 52624, 52625, 52626, 52627, 52628, 52462, 52630, 52631, 52632, 52633, 52634, 52463, 52636, 52637, 52638, 52639, 52464, 61339, 52465, 61340, 52466, 60606, 52467, 61317, 61342, 52468, 62925, 61343, 52469, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 52475, 52629, 52476, 52477, 52478, 62927, 52479, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 22048, 22049, 22050, 22051, 22052, 22053, 22054, 22055, 22056, 22057, 22058, 22059, 22060, 22061, 22062, 22063, 22064, 22065, 22066, 22067, 22068, 22069, 22070, 22071, 22072, 22073, 22074, 17511, 22076, 22077, 22078, 22079, 11840, 11841, 11842, 11843, 11844, 11845, 11846, 11847, 11848, 11849, 11850, 11851, 11852, 11853, 11854, 11855, 11856, 11857, 11858, 11859, 11860, 11861, 11862, 11863, 11864, 11865, 11866, 11867, 11868, 11869, 11870, 11871, 38243, 54338, 62916, 52796, 62914, 17513, 62932, 17514, 52635, 62937, 7840, 7841, 7842, 7843, 7844, 7845, 7846, 7847, 7848, 7849, 7850, 7851, 7852, 7853, 7854, 7855, 7856, 7857, 7858, 7859, 7860, 7861, 7862, 7863, 7864, 7865, 7866, 7867, 7868, 7869, 7870, 7871, 52797, 63779, 52798, 54342, 62935, 32929, 52774, 54343, 62915, 52775, 60595, 60604, 63788, 63792, 54344, 52454, 52779, 55072, 55073, 55074, 45052, 55076, 55077, 55078, 52776, 55080, 55081, 55082, 55083, 55084, 55085, 55086, 55087, 55088, 55089, 55090, 55091, 55092, 55093, 54345, 52458, 55096, 55097, 55098, 55099, 55100, 27067, 55102, 55103, 62938, 45024, 62941, 52777, 45025, 63799, 55535, 45026, 61322, 45027, 54346, 45028, 27068, 62939, 45029, 60597, 52778, 27070, 45030, 45031, 52795, 45032, 45033, 27069, 62940, 45034, 40616, 20352, 20353, 20354, 20355, 20356, 20357, 20358, 20359, 20360, 20361, 20362, 20363, 20364, 20365, 20366, 20367, 20368, 20369, 20370, 20371, 20372, 20373, 20374, 20375, 20376, 20377, 20378, 20379, 20380, 20381, 20382, 20383, 40614, 45055, 45041, 45042, 54349, 52455, 45043, 27071, 62942, 45044, 52611, 52781, 45045, 40608, 47927, 45046, 40609, 45047, 40610, 62930, 17512, 45048, 62933, 40611, 60598, 62943, 45049, 40612, 52782, 45050, 40613, 36832, 36833, 36834, 36835, 36836, 36837, 36838, 36839, 36840, 36841, 36842, 36843, 36844, 36845, 36846, 36847, 36848, 36849, 36850, 36851, 36852, 36853, 36854, 36855, 36856, 36857, 36858, 36859, 36860, 36861, 36862, 36863])


      # if not onbits == onbits36:
      #   print onbits - onbits36
      #   print onbits36 - onbits
      #   assert False

    # if self.iterationIdx == 48:
    #   onbits = set(numpy.nonzero(y)[0])
    #   onbits48 = set([1544,  2400,  2401,  2402,  2403,  2404,  2405,  2406,  2407,
    #     2408,  2409,  2410,  2411,  2412,  2413,  2414,  2415,  2416,
    #     2417,  2418,  2419,  2420,  2421,  2422,  2423,  2424,  2425,
    #     2426,  2427,  2428,  2429,  2430,  2431,  7117,  7420,  9344,
    #     9345,  9346,  9347,  9348,  9349,  9350,  9351,  9352,  9353,
    #     9354,  9355,  9356,  9357,  9358,  9359,  9360,  9361,  9362,
    #     9363,  9364,  9365,  9366,  9367,  9368,  9369,  9370,  9371,
    #     9372,  9373,  9374,  9375,  9495, 10684, 11851, 17525, 17857,
    #    20960, 20961, 20962, 20963, 20964, 20965, 20966, 20967, 20968,
    #    20969, 20970, 20971, 20972, 20973, 20974, 20975, 20976, 20977,
    #    20978, 20979, 20980, 20981, 20982, 20983, 20984, 20985, 20986,
    #    20987, 20988, 20989, 20990, 20991, 22064, 22624, 22625, 22626,
    #    22627, 22628, 22629, 22630, 22631, 22632, 22633, 22634, 22635,
    #    22636, 22637, 22638, 22639, 22640, 22641, 22642, 22643, 22644,
    #    22645, 22646, 22647, 22648, 22649, 22650, 22651, 22652, 22653,
    #    22654, 22655, 23808, 23809, 23810, 23811, 23812, 23813, 23814,
    #    23815, 23816, 23817, 23818, 23819, 23820, 23821, 23822, 23823,
    #    23824, 23825, 23826, 23827, 23828, 23829, 23830, 23831, 23832,
    #    23833, 23834, 23835, 23836, 23837, 23838, 23839, 27040, 27041,
    #    27042, 27043, 27044, 27045, 27046, 27047, 27048, 27049, 27050,
    #    27051, 27052, 27053, 27054, 27055, 27056, 27057, 27058, 27059,
    #    27060, 27061, 27062, 27063, 27064, 27065, 27066, 27067, 27068,
    #    27069, 27070, 27071, 27136, 27137, 27138, 27139, 27140, 27141,
    #    27142, 27143, 27144, 27145, 27146, 27147, 27148, 27149, 27150,
    #    27151, 27152, 27153, 27154, 27155, 27156, 27157, 27158, 27159,
    #    27160, 27161, 27162, 27163, 27164, 27165, 27166, 27167, 27392,
    #    27393, 27394, 27395, 27396, 27397, 27398, 27399, 27400, 27401,
    #    27402, 27403, 27404, 27405, 27406, 27407, 27408, 27409, 27410,
    #    27411, 27412, 27413, 27414, 27415, 27416, 27417, 27418, 27419,
    #    27420, 27421, 27422, 27423, 27520, 27521, 27522, 27523, 27524,
    #    27525, 27526, 27527, 27528, 27529, 27530, 27531, 27532, 27533,
    #    27534, 27535, 27536, 27537, 27538, 27539, 27540, 27541, 27542,
    #    27543, 27544, 27545, 27546, 27547, 27548, 27549, 27550, 27551,
    #    32096, 32097, 32098, 32099, 32100, 32101, 32102, 32103, 32104,
    #    32105, 32106, 32107, 32108, 32109, 32110, 32111, 32112, 32113,
    #    32114, 32115, 32116, 32117, 32118, 32119, 32120, 32121, 32122,
    #    32123, 32124, 32125, 32126, 32127, 32940, 32960, 32961, 32962,
    #    32963, 32964, 32965, 32966, 32967, 32968, 32969, 32970, 32971,
    #    32972, 32973, 32974, 32975, 32976, 32977, 32978, 32979, 32980,
    #    32981, 32982, 32983, 32984, 32985, 32986, 32987, 32988, 32989,
    #    32990, 32991, 35909, 38055, 39808, 39809, 39810, 39811, 39812,
    #    39813, 39814, 39815, 39816, 39817, 39818, 39819, 39820, 39821,
    #    39822, 39823, 39824, 39825, 39826, 39827, 39828, 39829, 39830,
    #    39831, 39832, 39833, 39834, 39835, 39836, 39837, 39838, 39839,
    #    40960, 40961, 40962, 40963, 40964, 40965, 40966, 40967, 40968,
    #    40969, 40970, 40971, 40972, 40973, 40974, 40975, 40976, 40977,
    #    40978, 40979, 40980, 40981, 40982, 40983, 40984, 40985, 40986,
    #    40987, 40988, 40989, 40990, 40991, 41092, 49792, 49793, 49794,
    #    49795, 49796, 49797, 49798, 49799, 49800, 49801, 49802, 49803,
    #    49804, 49805, 49806, 49807, 49808, 49809, 49810, 49811, 49812,
    #    49813, 49814, 49815, 49816, 49817, 49818, 49819, 49820, 49821,
    #    49822, 49823, 53984, 53985, 53986, 53987, 53988, 53989, 53990,
    #    53991, 53992, 53993, 53994, 53995, 53996, 53997, 53998, 53999,
    #    54000, 54001, 54002, 54003, 54004, 54005, 54006, 54007, 54008,
    #    54009, 54010, 54011, 54012, 54013, 54014, 54015, 55520, 55521,
    #    55522, 55523, 55524, 55525, 55526, 55527, 55528, 55529, 55530,
    #    55531, 55532, 55533, 55534, 55535, 55536, 55537, 55538, 55539,
    #    55540, 55541, 55542, 55543, 55544, 55545, 55546, 55547, 55548,
    #    55549, 55550, 55551, 56544, 56545, 56546, 56547, 56548, 56549,
    #    56550, 56551, 56552, 56553, 56554, 56555, 56556, 56557, 56558,
    #    56559, 56560, 56561, 56562, 56563, 56564, 56565, 56566, 56567,
    #    56568, 56569, 56570, 56571, 56572, 56573, 56574, 56575, 57056,
    #    57057, 57058, 57059, 57060, 57061, 57062, 57063, 57064, 57065,
    #    57066, 57067, 57068, 57069, 57070, 57071, 57072, 57073, 57074,
    #    57075, 57076, 57077, 57078, 57079, 57080, 57081, 57082, 57083,
    #    57084, 57085, 57086, 57087, 57824, 57825, 57826, 57827, 57828,
    #    57829, 57830, 57831, 57832, 57833, 57834, 57835, 57836, 57837,
    #    57838, 57839, 57840, 57841, 57842, 57843, 57844, 57845, 57846,
    #    57847, 57848, 57849, 57850, 57851, 57852, 57853, 57854, 57855,
    #    59680, 59681, 59682, 59683, 59684, 59685, 59686, 59687, 59688,
    #    59689, 59690, 59691, 59692, 59693, 59694, 59695, 59696, 59697,
    #    59698, 59699, 59700, 59701, 59702, 59703, 59704, 59705, 59706,
    #    59707, 59708, 59709, 59710, 59711, 60128, 60129, 60130, 60131,
    #    60132, 60133, 60134, 60135, 60136, 60137, 60138, 60139, 60140,
    #    60141, 60142, 60143, 60144, 60145, 60146, 60147, 60148, 60149,
    #    60150, 60151, 60152, 60153, 60154, 60155, 60156, 60157, 60158,
    #    60159, 60801, 61312, 61313, 61314, 61315, 61316, 61317, 61318,
    #    61319, 61320, 61321, 61322, 61323, 61324, 61325, 61326, 61327,
    #    61328, 61329, 61330, 61331, 61332, 61333, 61334, 61335, 61336,
    #    61337, 61338, 61339, 61340, 61341, 61342, 61343, 62928, 63424,
    #    63425, 63426, 63427, 63428, 63429, 63430, 63431, 63432, 63433,
    #    63434, 63435, 63436, 63437, 63438, 63439, 63440, 63441, 63442,
    #    63443, 63444, 63445, 63446, 63447, 63448, 63449, 63450, 63451,
    #    63452, 63453, 63454, 63455, 63787, 63808, 63809, 63810, 63811,
    #    63812, 63813, 63814, 63815, 63816, 63817, 63818, 63819, 63820,
    #    63821, 63822, 63823, 63824, 63825, 63826, 63827, 63828, 63829,
    #    63830, 63831, 63832, 63833, 63834, 63835, 63836, 63837, 63838, 63839])

    #   if not onbits == onbits48:
    #     print onbits - onbits48
    #     print onbits48 - onbits
    #     assert False

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
