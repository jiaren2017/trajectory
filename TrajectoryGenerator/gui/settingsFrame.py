import wx
import os
import configparser

class SettingsFrame(wx.Frame):

    def __init__(self, parent=None, title="Settings"):
        title = "Settings"
        size=(230,230)
        wx.Frame.__init__(self, None, wx.ID_ANY, title=title, size=size)

        this_directory = os.path.dirname(os.path.abspath(__file__))
        top_directory = os.path.split(this_directory)[0]
        self.config_path = os.sep.join([top_directory, "settings.ini"])

        self.init_UI()
        self.load_values()

        self.Centre()
        self.Show()

    def init_UI(self):
        flags = wx.ALIGN_CENTER_VERTICAL|wx.ALL

        panel = wx.Panel(self)
        mainSizer = wx.BoxSizer(wx.VERTICAL)

        sb = wx.StaticBox(panel, label="Export")
        exportSizer = wx.StaticBoxSizer(sb, wx.HORIZONTAL)
        self.tx_size = wx.StaticText(panel, label="Image-Size (x,y)")
        self.tc_size = wx.TextCtrl(panel)
        self.tx_dpi = wx.StaticText(panel, label="DPI")
        self.tc_dpi = wx.TextCtrl(panel)
        self.tx_scale = wx.StaticText(panel, label="Scale (N.A.)")
        self.tx_scale.SetToolTip('Currently not yet implemented!')
        self.tc_scale = wx.TextCtrl(panel)
        self.tc_scale.SetToolTip('Currently not yet implemented!')
        
        test = wx.FlexGridSizer(3, 2, 10, 10)
        test.AddMany([(self.tx_size, 1, wx.ALIGN_CENTER),(self.tc_size, 2, wx.EXPAND), \
                      (self.tx_dpi, 1, wx.ALIGN_CENTER), (self.tc_dpi, 2, wx.EXPAND), \
                      (self.tx_scale, 1, wx.ALIGN_CENTER), (self.tc_scale, 2, wx.EXPAND)])
        exportSizer.Add(test,1,flag=wx.ALL,border=5)
        self.btn_quit = wx.Button(panel, label="Save")

        mainSizer.Add(exportSizer, 1, flag=flags, border=5)
        mainSizer.Add(self.btn_quit, 0.2, flag=flags, border=5)

        self.Bind(wx.EVT_BUTTON, self.on_quit, self.btn_quit)
        # self.Bind(wx.EVT_BUTTON, self.on_quit, self.btn_quit)

        panel.SetSizer(mainSizer)

    def load_values(self):
        config = configparser.ConfigParser()
        config.read(self.config_path)
        self.tc_size.SetValue(config["Export"]["size"])
        self.tc_dpi.SetValue(config["Export"]["dpi"])
        self.tc_scale.SetValue(config["Export"]["scale"])

    def on_test(self, evt):
        print(self.GetSize())
        return

    def on_quit(self, evt):

        size = self.tc_size.GetValue().replace("(","").replace(")","")
        try:
            x, y = [int(el) for el in size.split(",")]
            dpi = int(self.tc_dpi.GetValue())
            scale = int(self.tc_scale.GetValue())
        except ValueError:
            wx.MessageBox('Invalid Setting-values', 'Error', \
                          wx.OK | wx.ICON_ERROR)
            return
        config = configparser.ConfigParser()
        config.read(self.config_path)
        config["Export"] = {"size": self.tc_size.GetValue(), \
                            "dpi": self.tc_dpi.GetValue(), \
                            "scale": self.tc_scale.GetValue()}
        with open(self.config_path, "w") as config_file:
            config.write(config_file)
        self.Close()

if __name__ == '__main__':
    ex = wx.App()
    SettingsFrame(None)
    ex.MainLoop()
