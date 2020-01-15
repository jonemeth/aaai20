

import os


baseDir = 'results'
if not os.path.exists(baseDir):
  os.makedirs(baseDir)


for ac in ['GVAE', 'MLVAE']:
  for dims in [ [2, 14], [8,8], [2, 6]]:
    dc = dims[0]
    ds = dims[1]

    for gs in [2, 5, 10, 20]:
     for mi in ['X_ZI', 'NONE']:
      for b1 in [1.0, 0.5, 0.25]:
       for b2 in [ 1.0, 2.0, 5.0, 2.5, 10.0, 20.0]:

        testDir = baseDir + '/results_{0}_{1:02d}_{2:02d}_{3:02d}_{4}_{5}_{6}'.format(ac, dc, ds, gs, mi, b1, b2)
        
        args = ''
        
        modeldir = testDir + '/saves'
        cfgfile = testDir + '/cfg.dat'


        if not os.path.exists(modeldir) or not os.path.exists(modeldir+'/iter.txt'):
          continue
        
        with open( modeldir + '/iter.txt' ) as f:
          iters = int(f.read()) 
          
        outfile = testDir + '/test_mnist_'+str(iters)+'_svm-ovo.txt'                                                                                       
        logfile = testDir + '/test_mnist_'+str(iters)+'_svm-ovo.log'                                                                                       

        if not os.path.exists(outfile):
          os.system( 'python3 test_classify_svm.py --decision_function_shape=ovo --experiment=mnist --modeldir '+modeldir+' --cfgfile '+cfgfile+' --outfile '+outfile+' '+args+' | tee '+logfile )

