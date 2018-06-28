import wx
import csv

from matplotlib.image import imread
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar


class PlotTraceTab(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.init_variables()
        self.init_plot()
        self.init_UI()


    def init_variables(self):
        self.cid = 0
        self.recording_trace = False


    def init_plot(self):
        self.dpi = 100
        self.fig = Figure((8.5,8.5), dpi=self.dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_xlabel("x")
        self.axes.set_ylabel("y", rotation="horizontal", labelpad=20)
        self.trace_plot, = self.axes.plot([],[],c="r")
        self.trace_marker, = self.axes.plot([],[], c="r", \
                                            marker="x", markersize=10)
        """ Als Workaround fuer den Home-Button notwendig....anpassen!"""
        self.axes.set_xlim((0,1200))
        self.axes.set_ylim((1200,0))
        self.fig.tight_layout()

    def init_UI(self):
        mainSizer = wx.BoxSizer(wx.VERTICAL)

        self.canvas = FigCanvas(self, -1, self.fig)
        mainSizer.Add(self.canvas, proportion=9, \
                      flag=wx.ALIGN_CENTER|wx.EXPAND)
        self.toolbar = NavigationToolbar(self.canvas)
        mainSizer.Add(self.toolbar, proportion=1, \
                      flag=wx.ALIGN_CENTER|wx.ALL|wx.EXPAND, border=5)

        self.SetSizer(mainSizer)
        self.Fit()


    def load_image(self, image_path):
        # self.fig.clf()
        self.axes.cla()
        self.axes.set_xlabel("x")
        self.axes.set_ylabel("y", rotation="horizontal", labelpad=20)
        self.trace_plot, = self.axes.plot([],[],c="r")
        self.trace_marker, = self.axes.plot([],[], c="r", \
                                            marker="x", markersize=10)
        img = imread(image_path)
        self.axes.imshow(img)
        self.fig.canvas.draw()


if __name__ == "__main__":
    class DemoFrame(wx.Frame):
        def __init__(self):
            wx.Frame.__init__(self, None, wx.ID_ANY,
                           "Notebook Test"
                            )
                            #,size=(330,700))
            panel = wx.Panel(self)
            self.notebook = wx.Notebook(panel)
            self.simTab = PlotTraceTab(self.notebook)
            test = SimSettingsTab(self.notebook)
            self.notebook.AddPage(self.simTab, "Simulation")
            self.notebook.AddPage(test, "Test")
            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(self.notebook, 1, wx.ALL|wx.EXPAND, 5)
            panel.SetSizer(sizer)
            self.Show()


        def onExit(self, evt):
            self.Destroy()

        def demo(self):
            self.image_path = '../maps/street_640x400_4goals.png'
            self.image_path = '/home/dev/Pictures/mensa_bearbeitet.png'
            # trace_data = [(0,0),(24,60),(206,60),(231,68),(243,85),(260,125),\
            #           (365,315),(375,325),(395,340),(430,340),(600,340)]
            # for pos in trace_data:
            #     self.simTab.simTraceTab.trace_add(pos[0], pos[1])
            # self.simTab.plot_trace()
            self.simTab.axes.set_xlim((0,600))
            self.simTab.axes.set_ylim((0,600))
            # self.simTab.axes.autoscale(True, "both")
            self.simTab.load_image(self.image_path)

        def onSize(self, evt):
            size = self.GetSize()
            print(size)
    app = wx.App()
    frame = DemoFrame()
    # frame.demo()
    app.MainLoop()
