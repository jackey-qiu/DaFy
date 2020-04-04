import os
import inspect
#use to locate the folder path
def local_func():
    return None
    
def module_path_locator(func=local_func):
    return os.path.dirname(os.path.abspath(inspect.getsourcefile(func)))        
