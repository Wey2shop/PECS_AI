import os, shutil, random, glob, warnings


cwd = os.getcwd()
print(cwd)
if os.path.basename(os.getcwd()) != 'train':
    os.chdir('dataset/train')
    print(cwd + " Most likely the Directory structure allready exists..")

if os.path.isdir('train/i want') is False:
    os.makedirs('train/i want')
    os.makedirs('train/biscuit')
    os.makedirs('train/milk')
    os.makedirs('train/choosing')
    os.makedirs('train/cake')
    
    os.makedirs('valid/i want')
    os.makedirs('valid/biscuit')
    os.makedirs('valid/milk')
    os.makedirs('valid/choosing')
    os.makedirs('valid/cake')
    
    os.makedirs('test/i want')
    os.makedirs('test/biscuit')
    os.makedirs('test/milk')
    os.makedirs('test/choosing')
    os.makedirs('test/cake')
    
    print(cwd + " Creating File structure and Moving Image data in to appropriate Directories.")

    for c in random.sample(glob.glob('*biscuit*'), 500):
        shutil.move(c, 'train/biscuit')
    
    for c in random.sample(glob.glob('*i_want*'), 500):
        shutil.move(c, 'train/i_want')
        
    for c in random.sample(glob.glob('*milk*'), 500):
        shutil.move(c, 'train/milk')        
    
    for c in random.sample(glob.glob('*choosing*'), 500):
        shutil.move(c, 'train/choosing')    
    
    for c in random.sample(glob.glob('*cake*'), 500):
        shutil.move(c, 'train/cake')

        
    
    for c in random.sample(glob.glob('*biscuit*'), 100):
        shutil.move(c, 'valid/biscuit')
    
    for c in random.sample(glob.glob('*i_want*'), 100):
        shutil.move(c, 'valid/i_want')
        
    for c in random.sample(glob.glob('*milk*'), 100):
        shutil.move(c, 'valid/milk')        
  
    for c in random.sample(glob.glob('*choosing*'), 100):
        shutil.move(c, 'valid/choosing')
  
    for c in random.sample(glob.glob('*cake*'), 100):
        shutil.move(c, 'valid/cake')
        
  
  
    for c in random.sample(glob.glob('*biscuit*'), 50):
        shutil.move(c, 'test/biscuit')
    
    for c in random.sample(glob.glob('*i_want*'), 50):
        shutil.move(c, 'test/i_want')
        
    for c in random.sample(glob.glob('*milk*'), 50):
        shutil.move(c, 'test/milk')        
  
    for c in random.sample(glob.glob('*choosing*'), 50):
        shutil.move(c, 'test/choosing')
  
    for c in random.sample(glob.glob('*cake*'), 50):
        shutil.move(c, 'test/cake')

os.chdir('../../')
print("Complete.. Going back to dir" + cwd)