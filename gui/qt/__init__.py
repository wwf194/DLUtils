
import sys, os
def ApplySystemScalingQt(Enable=True):
    if Enable: # UseSystemDPI
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    else: # Windows system don't apply any DPI scaling.
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    # app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

def GetScreenSize():
    from win32api import GetSystemMetrics
    # after DPI scaling / after system-level scaling.
    return (
        GetSystemMetrics(0), # width
        GetSystemMetrics(1), # height
    )

def SetWindowAlwaysOnTopWin32(hWindow):
    try:
        import win32gui, win32con
    except Exception:
        return False

    win32gui.SetWindowPos(
        hWindow, 
        win32con.HWND_TOPMOST, # window always on top
        0, # top left X 
        0, # top left Y
        100, # width
        100, # height
        win32con.SWP_SHOWWINDOW | win32con.SWP_NOSIZE # flags. SWP_NOSIZE: keep current width and height.
    )

def PrintScreenSize():
    import DLUtils
    Width, Height = GetScreenSize()
    DLUtils.Print("Screen(after system-level scaling) W: %d H: %d"%(Width, Height))

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.abspath(".."))
    X, Y = GetScreenSize()
    print(X, Y)
    PrintScreenSize()