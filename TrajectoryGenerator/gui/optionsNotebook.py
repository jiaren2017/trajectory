import wx

from gui.optionsTraceTab import OptionsTraceTab
from gui.optionsSettingsTab import OptionsSettingsTab
from gui.optionsResultsTab import OptionsResultsTab

class OptionsNotebook(wx.Notebook):

    def __init__(self, parent):
        wx.Notebook.__init__(self, parent, id=wx.ID_ANY, \
                             style=wx.BK_DEFAULT)
        self.init_UI()
        self.init_bindings()

    def init_UI(self):
        self.TraceTab = OptionsTraceTab(self)
        self.SettingsTab = OptionsSettingsTab(self)
        self.ResultsTab = OptionsResultsTab(self)
        self.AddPage(self.TraceTab, "Trace ")
        self.AddPage(self.SettingsTab, "Settings ")
        self.AddPage(self.ResultsTab, "Results ")


    def init_bindings(self):
        pass



if __name__ == "__main__":
    class DemoFrame(wx.Frame):
        def __init__(self):
            wx.Frame.__init__(self, None, wx.ID_ANY,
                           "Notebook Test"
                            )
                            #,size=(330,700))
            panel = wx.Panel(self)
            self.notebook = OptionsNotebook(panel)
            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(self.notebook, 1, wx.ALL|wx.EXPAND, 5)
            panel.SetSizer(sizer)
            self.Show()
        def onExit(self, evt):
            self.Destroy()

        def onSize(self, evt):
            size = self.GetSize()
            print(size)
    app = wx.App()
    frame = DemoFrame()
    app.MainLoop()
