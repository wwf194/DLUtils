
try:
    import win32con # pip install pywin32    
    import win32gui
    import psutil # pip install psutil
except Exception:
    pass
else:
    import DLUtils
    
    def ListNetworkInterfaceName():
        addrs = psutil.net_if_addrs()
        return addrs.keys()

    # def CloseWindow(WindowHandle):
    #     win32gui.PostMessage(WindowHandle, win32con.WM_CLOSE, 0, 0)
        
    def CloseWindow(WindowTitle):
        def winEnumHandler(handleWindow, ctx):
            if win32gui.IsWindowVisible(handleWindow):
                hex(handleWindow)
                TitleStr = win32gui.GetWindowText(handleWindow)
                if WindowTitle in TitleStr:
                    win32gui.PostMessage(handleWindow, win32con.WM_CLOSE, 0, 0)
                    DLUtils.print("closed window with title %s"%TitleStr)
        win32gui.EnumWindows(winEnumHandler, None)