from enum import Enum

class ArchitectureType(Enum):
  GVAE = 1

class RegularizationType(Enum):
  KL = 1
  

class MutualInformationType(Enum):
  NONE = 1
  X_ZI = 2

class AccumulationType(Enum):
  NONE = 0
  MLVAE = 1
  GVAE = 2


class cfg_base:
  def __init__(self):
    self.seed = 1
    self.mb_size = 100
    self.stat_moving_decay = 0.99
    self.train_drop_keep = 1.0
    self.sampleGridSize = 12  
    self.save_path = "./saves"
  
    self.mutualInformation = MutualInformationType.X_ZI
    self.regularization = RegularizationType.KL
    self.accumulation = AccumulationType.MLVAE
    self.architecture = ArchitectureType.GVAE
  
    self.gaussianEncoder = False
  
    self.target_size = 32
    self.color_channels = 1


    self.binary = True
    self.shiftMean = False  
    self.scatter = True
  
    self.initial_learning_rate = 0.001
    self.learning_beta1 = 0.9
    self.learning_beta2 = 0.999
  
    self.D_inner_iters = 3
  
    self.Istar = 0.200


  
################################################################################################################################


cfg = None


def config( experiment ):
    global cfg
    cfg = cfg_base()

    if 'mnist' == experiment:
        cfg.target_size = 32
        cfg.color_channels = 1

        cfg.initial_learning_rate = 0.001

        cfg.binary = True
        cfg.shiftMean = False
        cfg.numClasses = 10
        
        cfg.scatter = True

        cfg.mb_size = 100
        cfg.groupSize = 2
        cfg.max_it = 20000
        
        
        cfg.accumulation = AccumulationType.MLVAE
        cfg.regularization = RegularizationType.KL
        cfg.mutualInformation = MutualInformationType.X_ZI
        cfg.architecture = ArchitectureType.GVAE
        cfg.gaussianEncoder = True
        cfg.decoderVariance = False

    else:
        print('Unknown experiment')


