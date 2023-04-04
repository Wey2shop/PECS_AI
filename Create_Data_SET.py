import os, shutil, random, glob, warnings


cwd = os.getcwd()
print(cwd)
if os.path.basename(os.getcwd()) != 'train':
    os.chdir('dataset/train')
        print(cwd + " Most likely the Directory structure allready exists..")

if os.path.isdir('train/i_want') is False:
    os.makedirs('train/i_want')
    os.makedirs('train/biscuit')
    os.makedirs('valid/i_want')
    os.makedirs('valid/biscuit')
    os.makedirs('test/i_want')
    os.makedirs('test/biscuit')
    print(cwd + " Creating File structure and Moving Image data in to appropriate Directories.")

    for c in random.sample(glob.glob('*biscuit*'), 500):
        shutil.move(c, 'train/biscuit')
    
    for c in random.sample(glob.glob('*i_want*'), 500):
        shutil.move(c, 'train/i_want')
		
		
    
    for c in random.sample(glob.glob('*biscuit*'), 100):
        shutil.move(c, 'valid/biscuit')
    
    for c in random.sample(glob.glob('*i_want*'), 100):
        shutil.move(c, 'valid/i_want')
		
		
    
    for c in random.sample(glob.glob('*biscuit*'), 50):
        shutil.move(c, 'test/biscuit')
    
    for c in random.sample(glob.glob('*i_want*'), 50):
        shutil.move(c, 'test/i_want')

os.chdir('../../')
print(cwd)