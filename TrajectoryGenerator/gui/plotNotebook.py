import wx

from gui.plotTraceTab import PlotTraceTab
from gui.plotResultsTab import PlotResultsTab

class PlotNotebook(wx.Notebook):

    def __init__(self, parent):
        wx.Notebook.__init__(self, parent, id=wx.ID_ANY, \
                             style=wx.BK_DEFAULT)
        self.init_UI()
        self.init_bindings()

    def init_UI(self):
        self.TracePlot = PlotTraceTab(self)
        self.ResultsPlot = PlotResultsTab(self)
        self.AddPage(self.TracePlot, "Trace ")
        self.AddPage(self.ResultsPlot, "Results")


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
            self.notebook = PlotNotebook(panel)
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
