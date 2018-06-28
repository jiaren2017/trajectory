import os
import wx
import configparser

class OptionsSettingsTab(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, id=wx.ID_ANY)
        this_directory = os.path.dirname(os.path.abspath(__file__))
        top_directory = os.path.split(this_directory)[0]
        self.config_path = os.sep.join([top_directory, "settings.ini"])

        self.init_UI()
        self.load_defaults()


    def init_UI(self):
        outer_box = wx.BoxSizer(wx.VERTICAL)
        sizer = wx.GridBagSizer(5,5)

        """ Definitions """
        """TODO: Tooltips """
        pre_noise_label = u'Pre-\N{GREEK SMALL LETTER SIGMA}:'
        tx_pre_noise = wx.StaticText(self, label=pre_noise_label)
        tt_pre = "Standard Deviation of Noise added to Trace-Postitions"
        tx_pre_noise.SetToolTip(tt_pre)
        self.tc_pre_noise = wx.TextCtrl(self)

        post_noise_label = u'Post-\N{GREEK SMALL LETTER SIGMA}:'
        tx_post_noise = wx.StaticText(self, label=post_noise_label)
        tt_post = "Standard Deviation of Noise added to each resulting"+\
                  "Trace-Value"
        tx_post_noise.SetToolTip(tt_post)
        self.tc_post_noise = wx.TextCtrl(self)


        r""" Interpolation"""
        line1 = wx.StaticLine(self)
        self.rb_interp = wx.RadioButton(self, label="Interpolation",\
                                          style=wx.RB_GROUP)
        self.tx_interp_kind = wx.StaticText(self, label="Method: ")
        tt_interp = "Specifies the kind of interpolation for values" +\
                    "between the trace-points."
        self.tx_interp_kind.SetToolTip(tt_interp)
        self.interps = ["cubic", "quadratic", "linear", "slinear", \
                   "nearest", "zero", "univariate"]
        self.cb_interp_kind = wx.ComboBox(self, choices=self.interps, \
                                     style=wx.CB_READONLY)

        self.tx_interp_factor = wx.StaticText(self, label="Factor:")
        tt_factor = "Factor by which the resulting walk has more points"+\
                    " than the original trace."
        self.tx_interp_factor.SetToolTip(tt_factor)
        self.tc_interp_factor = wx.TextCtrl(self)

        self.interp_widgets = [self.tx_interp_kind, self.cb_interp_kind, \
                               self.tx_interp_factor, self.tc_interp_factor]

        """ Simulation"""
        self.rb_sim = wx.RadioButton(self, label="Simulation")
        self.tx_sim_method = wx.StaticText(self, label="Method: ")
        tt_sim = "Specifies the kind of interpolation for values" +\
                    "between the trace-points."
        self.tx_sim_method.SetToolTip(tt_sim)
        self.sims = ["Seek", "Wander"]
        self.cb_sim = wx.ComboBox(self, choices=self.sims, \
                                     style=wx.CB_READONLY)
        self.tx_sim_mass = wx.StaticText(self, label="Mass:")
        self.tc_sim_mass = wx.TextCtrl(self)

        self.tx_sim_dt = wx.StaticText(self, label="dt:")
        self.tc_sim_dt = wx.TextCtrl(self)

        self.tx_sim_vmax = wx.StaticText(self, label="vMax:")
        self.tc_sim_vmax = wx.TextCtrl(self)

        self.tx_sim_rgoal = wx.StaticText(self, label="rGoal:")
        self.tc_sim_rgoal = wx.TextCtrl(self)


        self.sim_widgets = [self.tx_sim_method, self.cb_sim,\
                            self.tx_sim_mass, self.tc_sim_mass,\
                            self.tx_sim_dt, self.tc_sim_dt,\
                            self.tx_sim_vmax, self.tc_sim_vmax,\
                            self.tx_sim_rgoal, self.tc_sim_rgoal]

        r""" Settings """
        line2 = wx.StaticLine(self)
        tx_nr = wx.StaticText(self, label="# Runs:")
        self.tc_nr = wx.TextCtrl(self)
        tx_label = wx.StaticText(self, label="Label:")
        self.tc_label = wx.TextCtrl(self)

        self.btn_run = wx.Button(self, label="Run")
        #
        self.Bind(wx.EVT_BUTTON, self.save_defaults, self.btn_run)

        std_flags = wx.ALIGN_CENTER_VERTICAL|wx.ALL
        exp_flags = std_flags|wx.EXPAND

        """ Sizer """
        sizer.Add(tx_pre_noise, pos=(0,0), flag=std_flags, border=5)
        sizer.Add(self.tc_pre_noise, pos=(0,1), flag=std_flags, border=5)
        sizer.Add(tx_post_noise, pos=(1,0), flag=std_flags, border=5)
        sizer.Add(self.tc_post_noise, pos=(1,1), flag=std_flags, border=5)
        sizer.Add(line1, pos=(2,0), span=(1,3), flag=exp_flags, border=5)

        sizer.Add(self.rb_interp, pos=(3,0), span=(1,2),
                  flag=exp_flags, border=5)
        sizer.Add(self.tx_interp_kind, pos=(4,1), flag=std_flags, border=5)
        sizer.Add(self.cb_interp_kind, pos=(4,2), flag=std_flags, border=5)
        sizer.Add(self.tx_interp_factor, pos=(5,1), flag=std_flags, border=5)
        sizer.Add(self.tc_interp_factor, pos=(5,2), flag=std_flags, border=5)

        sizer.Add(self.rb_sim, pos=(6,0), span=(1,2), \
                  flag=exp_flags, border=5)
        sizer.Add(self.tx_sim_method, pos=(7,1), flag=std_flags, border=5)
        sizer.Add(self.cb_sim, pos=(7,2), flag=std_flags, border=5)
        sizer.Add(self.tx_sim_mass, pos=(8,1), flag=std_flags, border=5)
        sizer.Add(self.tc_sim_mass, pos=(8,2), flag=std_flags, border=5)
        sizer.Add(self.tx_sim_dt, pos=(9,1), flag=std_flags, border=5)
        sizer.Add(self.tc_sim_dt, pos=(9,2), flag=std_flags, border=5)
        sizer.Add(self.tx_sim_vmax, pos=(10,1), flag=std_flags, border=5)
        sizer.Add(self.tc_sim_vmax, pos=(10,2), flag=std_flags, border=5)
        sizer.Add(self.tx_sim_rgoal, pos=(11,1), flag=std_flags, border=5)
        sizer.Add(self.tc_sim_rgoal, pos=(11,2), flag=std_flags, border=5)

        sizer.Add(line2, pos=(12,0), span=(1,3), flag=exp_flags, border=5)
        sizer.Add(tx_nr, pos=(13,0), flag=std_flags, border=5)
        sizer.Add(self.tc_nr, pos=(13,1), flag=std_flags, border=5)
        sizer.Add(tx_label, pos=(14,0), flag=std_flags, border=5)
        sizer.Add(self.tc_label, pos=(14,1), flag=std_flags, border=5)
        sizer.Add(self.btn_run, pos=(15,1), flag=std_flags, border=5)

        outer_box.Add(sizer, 1, wx.ALL, border=10)
        self.SetSizer(outer_box)


        self.Bind(wx.EVT_RADIOBUTTON, self.on_choose_interp, \
                  self.rb_interp)
        self.Bind(wx.EVT_RADIOBUTTON, self.on_choose_sim, \
                  self.rb_sim)
        self.on_choose_interp(None)

    def on_choose_interp(self, evt):
        for widget in self.interp_widgets:
            widget.Enable()
        for widget in self.sim_widgets:
            widget.Disable()

    def on_choose_sim(self, evt):
        for widget in self.interp_widgets:
            widget.Disable()
        for widget in self.sim_widgets:
            widget.Enable()

    def collect_config(self):
        config = {}
        try:
            config["pre_noise"] = float(self.tc_pre_noise.GetValue())
            config["post_noise"] = float(self.tc_post_noise.GetValue())
            if self.rb_interp.GetValue():
                config["method"] = "Interpolation"
                config["kind"] = self.cb_interp_kind.GetStringSelection()
                config["factor"] = float(self.tc_interp_factor.GetValue())
            else:
                config["method"] = "Simulation"
                config["kind"] = self.cb_sim.GetStringSelection()
                config["mass"] = float(self.tc_sim_mass.GetValue())
                config["dt"] = float(self.tc_sim_dt.GetValue())
                config["vmax"] = float(self.tc_sim_vmax.GetValue())
                config["rgoal"] = float(self.tc_sim_rgoal.GetValue())
            # else:
            #     msg = "Something went terribly wrong...."
            #     raise ValueError(msg)
            config["nr_runs"] = int(self.tc_nr.GetValue())
            config["label"] = self.tc_label.GetValue()
            return config

        except:
            msg = "Could not read config-values from Settings-Tab."
            raise ValueError(msg)

    def load_defaults(self):
        config = configparser.ConfigParser()
        config.read(self.config_path)
        self.tc_pre_noise.SetValue(str(config["Defaults"]["pre_noise"]))
        self.tc_post_noise.SetValue(str(config["Defaults"]["post_noise"]))
        if config["Generation"]["method"] == "Interpolation":
            self.rb_interp.SetValue(True)
            self.rb_sim.SetValue(False)
            self.cb_interp_kind.SetSelection(self.interps.index(config["Generation"]["kind"]))
            self.tc_interp_factor.SetValue(str(int(float(config["Generation"]["factor"]))))
        elif config["Generation"]["method"] == "Simulation":
            self.rb_interp.SetValue(False)
            self.rb_sim.SetValue(True)
            self.cb_sim.SetSelection(self.sims.index(config["Generation"]["kind"]))
            self.tc_sim_mass.SetValue(str(config["Generation"]["mass"]))
            self.tc_sim_dt.SetValue(str(config["Generation"]["dt"]))
            self.tc_sim_vmax.SetValue(str(config["Generation"]["vmax"]))
            self.tc_sim_rgoal.SetValue(str(config["Generation"]["rgoal"]))

        self.tc_nr.SetValue(config["Defaults"]["nr_runs"])
        self.tc_label.SetValue(config["Defaults"]["label"])


    def save_defaults(self, evt):
        r"""Write Defaults to .ini file.

        Bound directly to the run-Button, but the evt.Skip() should guarantee that
        all other functions are called as well.
        """
        valid_config = self.collect_config()
        cp = configparser.ConfigParser()
        cp.read(self.config_path)
        cp["Defaults"]["pre_noise"] = str(valid_config["pre_noise"])
        cp["Defaults"]["post_noise"] = str(valid_config["post_noise"])
        cp["Defaults"]["nr_runs"] = str(valid_config["nr_runs"])
        cp["Defaults"]["label"] = valid_config["label"]
        if valid_config["method"] == "Interpolation":
            cp["Generation"]["method"] = "Interpolation"
            cp["Generation"]["kind"] = valid_config["kind"]
            cp["Generation"]["factor"] = str(valid_config["factor"])
        elif valid_config["method"] == "Simulation":
            cp["Generation"]["method"] = "Simulation"
            cp["Generation"]["kind"] = valid_config["kind"]
            cp["Generation"]["mass"] = str(valid_config["mass"])
            cp["Generation"]["dt"] = str(valid_config["dt"])
            cp["Generation"]["vmax"] = str(valid_config["vmax"])
            cp["Generation"]["rgoal"] = str(valid_config["rgoal"])
        with open(self.config_path, "w") as config_file:
            cp.write(config_file)
        evt.Skip()

if __name__ == "__main__":
    class DemoFrame(wx.Frame):
        def __init__(self):
            wx.Frame.__init__(self, None, wx.ID_ANY,
                           "Notebook Test"
                            # )
                            ,size=(330,700))
            panel = wx.Panel(self)
            self.notebook = wx.Notebook(panel)
            self.SetTab = OptionsSettingsTab(self.notebook)
            self.notebook.AddPage(self.SetTab, "Settings")
            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(self.notebook, 1, wx.ALL|wx.EXPAND, 5)
            panel.SetSizer(sizer)
            self.Centre()
            self.Show()
            # self.Bind(wx.EVT_COMBOBOX, self.onTest, self.SetTab.cb_smoother)
            # self.Bind(wx.EVT_COMBOBOX, self.onCBSwitch, \
            #           self.SetTab.cb_smoother)

            # self.Bind(wx.EVT_BUTTON, self.onTest, \
            #           self.SetTab.btn_run)
        def onCBSwitch(self, evt):
            print(evt.GetString())


        def onExit(self, evt):
            self.Destroy()

        def onSize(self, evt):
            size = self.GetSize()

        def onTest(self, evt):
            res = self.SetTab.on_collect_config()
            for key in res:
                print(key,' - ',res[key])
            
    app = wx.App()
    frame = DemoFrame()
    app.MainLoop()
