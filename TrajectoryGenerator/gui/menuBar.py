import wx

class MenuBar(wx.MenuBar):
    def __init__(self, *args, **kwargs):
        super(MenuBar, self).__init__(*args, **kwargs)

        """ Main-Menu """
        self.sim_menu = wx.Menu()
        self.sim_set = self.sim_menu.Append(wx.ID_ANY, \
                                               "&Settings", \
                                               "Edit basic Settings")
        # self.sim_export = self.sim_menu.Append(wx.ID_ANY, \
        #                                        "Save Simulation-Settings", \
        #                                        "Export World-Image")
        self.sim_menu.AppendSeparator()
        self.sim_quit = self.sim_menu.Append(wx.ID_EXIT, "Quit",
                                              "Quit SillyWalks")

        """ Image-Menu """
        self.im_menu = wx.Menu()
        self.im_load = self.im_menu.Append(wx.ID_ANY, \
                                             "Load Background", \
                                             "Load a new Background-Image")
        self.im_trace_save = self.im_menu.Append(wx.ID_ANY, \
                                             "Save Trace-Plot", \
                                             "Save Trace-Plot to file.")
        self.im_res_save = self.im_menu.Append(wx.ID_ANY, \
                                             "Save Results-Plot", \
                                             "Save Results-Plot to file.")

        self.Append(self.sim_menu, "&Simulation")
        self.Append(self.im_menu, "&Image")

        # self.sim_menu.Disable()
