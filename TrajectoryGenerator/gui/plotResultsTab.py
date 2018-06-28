import wx
import csv
import numpy as np

from matplotlib.image import imread
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar


class PlotResultsTab(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        # self.init_variables()
        self.init_plot()
        self.init_UI()

    def init_plot(self):
        self.dpi = 100
        self.fig = Figure((8.5,8.5), dpi=self.dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_xlabel("x")
        self.axes.set_ylabel("y", rotation="horizontal", labelpad=20)
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


    def add_walk(self, x, y):
        new_plot, = self.axes.plot(x,y)
        self.fig.canvas.draw()

    def load_image(self, image_path):
        self.axes.cla()
        self.axes.set_xlabel("x")
        self.axes.set_ylabel("y", rotation="horizontal", labelpad=20)
        img = imread(image_path)
        self.axes.imshow(img)
        self.fig.canvas.draw()

    def get_trace_data(self, idx):
        try:
            x = self.axes.lines[idx].get_xdata()
            y = self.axes.lines[idx].get_ydata()
            return np.vstack((x,y))
        except:
            print("Error in get_trace_data: ",idx)
    # def on_export_selected(self, evt):
    #     selection = self.resTracePanel.walk_get_selected()
    #     if len(selection) > 0:
    #         for idx in selection:
    #             x = self.axes.lines[idx].get_xdata()
    #             y = self.axes.lines[idx].get_ydata()
    #             label = self.resTracePanel.walk_read(idx)
    #             print("X: ",x)
    #             print("Y: ",y)
    #             print("Label: ", label)
    #     print(self.resTracePanel.index)

    # def on_export_all(self, evt):
    #     if self.resTracePanel.index > 0:
    #         path = self.get_export_path()
    #         try:
    #             with open(path, "w") as f:
    #                 writer = csv.writer(f, delimiter=",")
    #                 for idx in range(self.resTracePanel.index):
    #                     x = self.axes.lines[idx].get_xdata()
    #                     y = self.axes.lines[idx].get_ydata()
    #                     label = self.resTracePanel.walk_read(idx)
    #                 for pos in zip(x,y):
    #                     writer.writerow(pos)
    #         except IOError:
    #             print("Could not write to file: ",pathname)


    # def get_export_path(self):
    #     wc = "CSV Files (*.csv)|*.csv|TXT Files (*.txt)|*.txt"
    #     wc += "|All Files|*.*;*"
    #     with wx.FileDialog(self, "Save Walks", wildcard=wc, \
    #                        style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT) as fd:
    #         if fd.ShowModal() == wx.ID_CANCEL:
    #             return
    #         return fd.GetPath()

    def plot_remove_selected(self, selection):
        if len(selection) > 0:
            for idx in selection[::-1]:
                self.axes.lines[idx].remove()
        self.fig.canvas.draw()

    def plot_clear_all(self):
        for line in self.axes.lines[::-1]:
            line.remove()
        self.fig.canvas.draw()

    def init_bindings(self):
        self.Bind(wx.EVT_BUTTON, self.on_remove_selected, \
                  self.resTracePanel.btn_rem)
        self.Bind(wx.EVT_BUTTON, self.on_export_selected, \
                  self.resTracePanel.btn_export)
        self.Bind(wx.EVT_BUTTON, self.on_clear_all, \
                  self.resTracePanel.btn_clear)

if __name__ == "__main__":
    import numpy as np
    class DemoFrame(wx.Frame):
        def __init__(self):
            wx.Frame.__init__(self, None, wx.ID_ANY,
                           "Notebook Test"
                            )
                            #,size=(330,700))
            panel = wx.Panel(self)
            self.notebook = wx.Notebook(panel)
            self.resTab = ResTab(self.notebook)
            self.notebook.AddPage(self.resTab, "Test")
            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(self.notebook, 1, wx.ALL|wx.EXPAND, 5)
            panel.SetSizer(sizer)
            self.Show()
        def onExit(self, evt):
            self.Destroy()

        def demo(self):
            x = np.arange(1,100)
            y1 = 0.1*x
            y2 = 1./x**2
            y3 = np.log(x)
            y4 = np.sin(x**2)
            self.resTab.add_walk(x,y1,'Yo')
            self.resTab.add_walk(x,y2,'Mama')
            self.resTab.add_walk(x,y3,'So')
            self.resTab.add_walk(x,y4,'Fat!')
        def onSize(self, evt):
            size = self.GetSize()
            print(size)
    app = wx.App()
    frame = DemoFrame()
    frame.demo()
    app.MainLoop()
