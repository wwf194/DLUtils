import win32con    
import win32gui

def CloseWindow(WindowHandle):
    win32gui.PostMessage(WindowHandle, win32con.WM_CLOSE, 0, 0)
    
import psutil
def ListNetworkInterfaceName():
    addrs = psutil.net_if_addrs()
    return addrs.keys()