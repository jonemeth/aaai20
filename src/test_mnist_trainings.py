

import os

baseDir = 'results'
if not os.path.exists(baseDir):
  os.makedirs(baseDir)


for ac in ['MLVAE']:
  for dims in [ [2, 14] ]:
    dc = dims[0]
    ds = dims[1]

    for gs in [2]:
     for mi in ['X_ZI', 'NONE']:
      for b1 in [1.0]:
       for b2 in [1.0]:
        
        outdir = baseDir + '/results_{0}_{1:02d}_{2:02d}_{3:02d}_{4}_{5}_{6}'.format(ac, dc, ds, gs, mi, b1, b2)
        args = '--ac '+ac    +' --dc '+str(dc)    +' --ds '+str(ds)    +' --gs '+str(gs)    +' --mi ' + mi+' --b1 ' + str(b1)+' --b2 ' + str(b2)    +' --outdir '+outdir

        modeldir = outdir + '/saves'
        if os.path.exists(modeldir+'/iter.txt'):
          with open( modeldir + '/iter.txt' ) as f:
            iters = int(f.read()) 
          
          if iters == 20000:
            continue
        
        if not os.path.exists(outdir):
          os.makedirs( outdir )

        os.system( 'python3 main_train.py --experiment=mnist '+args+' | tee '+outdir+'/LOG' )
