import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import configparser
from argparse import ArgumentParser

import wx
import cv2
import h5py

from gui.menuBar import MenuBar
from gui.plotNotebook import PlotNotebook
from gui.optionsNotebook import OptionsNotebook
from gui.settingsFrame import SettingsFrame

from mapCreation import image2array
from validation import validate_trace
from noiseGeneration import add_simple_noise
from walker.interpolatedWalk import interpolated_walk
from walker.simulatedWalk import SimulatedWalker

class TrajectoryGeneratorGui(wx.Frame):

    def __init__(self, parent=None, title="TrajectoryGenerator"):
        title = "TrajectoryGenerator"
        size=(1190,1010)
        wx.Frame.__init__(self, None, wx.ID_ANY, title=title, size=size)

        self.init_variables()
        self.init_menu()
        self.init_UI()
        self.init_bindings()

    def init_variables(self):
        self.image_path = ""
        self.recording_trace = False
        self.cid = -1
        this_directory = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.sep.join([this_directory, "settings.ini"])
        self.pos_dict = {}
        self.vel_dict = {}
        self.acc_dict = {}

    def init_menu(self):
        self.menubar = MenuBar()
        self.SetMenuBar(self.menubar)
        # self.CreateStatusBar()

    def init_UI(self):
        mainPanel = wx.Panel(self)
        mainSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.pNB = PlotNotebook(mainPanel)
        self.oNB = OptionsNotebook(mainPanel)
        mainSizer.Add(self.pNB, proportion=4, \
                      flag=wx.ALL|wx.EXPAND, border=5)
        mainSizer.Add(self.oNB, proportion=1, \
                      flag=wx.ALL|wx.EXPAND,border=5)
        mainPanel.SetSizer(mainSizer)
        self.Centre()
        self.Show()

    def init_bindings(self):
        """ Menu """
        self.Bind(wx.EVT_MENU, self.on_settings, self.menubar.sim_set)
        self.Bind(wx.EVT_MENU, self.on_exit, self.menubar.sim_quit)

        self.Bind(wx.EVT_MENU, self.on_load_images, \
                  self.menubar.im_load)

        self.Bind(wx.EVT_MENU, self.on_export_trace_plot, \
                  self.menubar.im_trace_save)
        self.Bind(wx.EVT_MENU, self.on_export_results_plot, \
                  self.menubar.im_res_save)

        """ Buttons """
        """ --- Trace-Tab """
        self.Bind(wx.EVT_BUTTON, self.on_toggle_recording, \
                  self.oNB.TraceTab.btn_record)
        self.Bind(wx.EVT_BUTTON, self.on_clear_trace, \
                  self.oNB.TraceTab.btn_clear)
        self.Bind(wx.EVT_BUTTON, self.on_trace_add_new, \
                  self.oNB.TraceTab.btn_add)
        self.Bind(wx.EVT_BUTTON, self.on_rem_trace_selection, \
                  self.oNB.TraceTab.btn_rem)
        self.Bind(wx.EVT_BUTTON, self.on_load_trace, \
                  self.oNB.TraceTab.btn_load)
        self.Bind(wx.EVT_BUTTON, self.on_save_trace, \
                  self.oNB.TraceTab.btn_save)

        """ --- Settings-Tab """
        self.Bind(wx.EVT_BUTTON, self.on_run_simulation, \
                  self.oNB.SettingsTab.btn_run)

        """ --- Results-Tab """
        self.Bind(wx.EVT_BUTTON, self.on_rem_results_selection, \
                  self.oNB.ResultsTab.btn_rem)
        self.Bind(wx.EVT_BUTTON, self.on_select_results_all, \
                  self.oNB.ResultsTab.btn_select_all)
        self.Bind(wx.EVT_BUTTON, self.on_rem_results_all, \
                  self.oNB.ResultsTab.btn_clear)
        self.Bind(wx.EVT_BUTTON, self.on_export_selected, \
                  self.oNB.ResultsTab.btn_export)

        """ Events """
        self.Bind(wx.EVT_LIST_END_LABEL_EDIT, self.on_reload_plot, \
                  self.oNB.TraceTab.trace)
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_stop_recording, \
                  self.pNB)
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_stop_recording, \
                  self.oNB)

    def on_select_results_all(self, evt):
        self.oNB.ResultsTab.walk_select_all()

    def on_export_selected(self, evt):
        selection = self.oNB.ResultsTab.walk_get_selected()
        write_mode = "w"
        if len(selection) == 0:
            wx.MessageBox("No Trace selected...", "Warning", \
                          wx.OK|wx.ICON_ERROR)
            return
        wc = "HDF5-File (*.hdf5)|*.hdf5|"
        wc += "All Files|*.*;*"
        with wx.FileDialog(self, "Export Traces", wildcard=wc, \
                           style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return
            pathname = fd.GetPath()
        count = 0
        try:
            with h5py.File(pathname, write_mode) as f:
                for index in selection:
                    label = self.oNB.ResultsTab.walk_read(index)
                    if label not in f.keys():
                        f.create_group(label)
                    grp = f[label].create_group(str(len(f[label])+1))
                    grp.create_dataset("Positions", data=self.pos_dict[index])
                    grp.create_dataset("Velocity", data=self.vel_dict[index])
                    grp.create_dataset("Acceleration", data=self.acc_dict[index])
        except OSError:
            msg = "File already open...CLOSE THE DAMN READER!"
            raise OSError(msg)

    def on_trace_add_new(self, evt):
        self.oNB.TraceTab.trace_add(0,0)
        wx.CallAfter(self.plot_trace)

    def on_rem_trace_selection(self, evt):
        self.oNB.TraceTab.trace_remove_selected()
        wx.CallAfter(self.plot_trace)

    def on_rem_results_all(self, evt):
        self.oNB.ResultsTab.walk_clear()
        self.pNB.ResultsPlot.plot_clear_all()
        self.pos_dict = {}
        self.vel_dict = {}
        self.acc_dict = {}

        # wx.CallAfter(self.plot_trace)

    def on_rem_results_selection(self, evt):
        selection = self.oNB.ResultsTab.walk_get_selected()
        self.oNB.ResultsTab.walk_remove_selected(selection)
        self.pNB.ResultsPlot.plot_remove_selected(selection)
        for key in selection:
            self.pos_dict.pop(key, None)
            self.vel_dict.pop(key, None)
            self.acc_dict.pop(key, None)
        i = 0
        for key in sorted(self.pos_dict.keys()):
            self.pos_dict[i] = self.pos_dict.pop(key)
            self.vel_dict[i] = self.vel_dict.pop(key)
            self.acc_dict[i] = self.acc_dict.pop(key)
            i += 1

    def on_clear_trace(self, evt):
        self.oNB.TraceTab.trace_clear()
        self.pNB.TracePlot.trace_plot.set_xdata([])
        self.pNB.TracePlot.trace_plot.set_ydata([])
        self.pNB.TracePlot.trace_marker.set_xdata([])
        self.pNB.TracePlot.trace_marker.set_ydata([])
        wx.CallAfter(self.pNB.TracePlot.fig.canvas.draw)


    def on_stop_recording(self, evt):
        r"""Stops recording a Trace when the notebook-tab changed"""
        if evt.GetOldSelection() == 0:
            try:
                self.recording_trace = False
                self.pNB.TracePlot.fig.canvas.mpl_disconnect(self.cid)
                label = "Record Trace"
                self.oNB.TraceTab.btn_record.SetLabel(label)
            except:
                # Nothing to do...
                pass

    def on_reload_plot(self, evt):
        wx.CallAfter(self.plot_trace)

    def on_toggle_recording(self, evt):
        if not self.recording_trace:
            self.recording_trace = True
            event = "button_press_event"
            self.cid = self.pNB.TracePlot.fig.canvas.mpl_connect(event,\
                                                self.on_pick_recording)
            self.oNB.TraceTab.btn_record.SetLabel("Stop")
        else:
            self.recording_trace = False
            self.pNB.TracePlot.fig.canvas.mpl_disconnect(self.cid)
            label = "Record Trace"
            self.oNB.TraceTab.btn_record.SetLabel(label)


    def on_pick_recording(self, evt):
        if evt.xdata != None and evt.ydata != None:
            x, y = int(evt.xdata), int(evt.ydata)
            self.oNB.TraceTab.trace_add(x, y)
            self.plot_trace()

    def plot_trace(self):
        x, y = self.oNB.TraceTab.trace_read()
        if False in (x,y):
            wx.MessageBox('Invalid Values in Trace-Table!', 'Error', \
                          wx.OK | wx.ICON_ERROR)
            return
        else:
            self.pNB.TracePlot.trace_plot.set_xdata(x)
            self.pNB.TracePlot.trace_plot.set_ydata(y)
            self.pNB.TracePlot.trace_marker.set_xdata(x[-1])
            self.pNB.TracePlot.trace_marker.set_ydata(y[-1])
        self.pNB.TracePlot.canvas.draw()


    def on_run_simulation(self, evt):
        # try:
        config = self.oNB.SettingsTab.collect_config()
        x, y = self.oNB.TraceTab.trace_read()
        config["x"] = x
        config["y"] = y
        if len(x) < 4 and config["method"] == "Interpolation":
            wx.MessageBox('Not enough Values for Interpolation!', 'Error', \
                          wx.OK | wx.ICON_ERROR)
            return
        config["path"] = self.image_path
        for i in range(config["nr_runs"]):
            pos, vel, acc = generate_walk(config)
            self.pNB.ResultsPlot.add_walk(pos[:,0], pos[:,1])
            index = self.oNB.ResultsTab.add_walk(config["label"])
            self.pos_dict[index] = pos
            self.vel_dict[index] = vel
            self.acc_dict[index] = acc
        # except ValueError:
        #     msg = "Invalid-Values in Settings tab:"
        #     print(msg)

    def on_load_images(self, evt):
        wc = "PNG Images (*.png)|*.png|JPG Images (*.jpg,*.jpeg)"
        wc += "|*.jpeg;*.jpeg|All Files|*.*;*"
        with wx.FileDialog(self, "Open World-Image",wildcard=wc, \
                           style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return
            self.image_path = fd.GetPath()
            try:
                self.pNB.TracePlot.load_image(self.image_path)
                self.pNB.ResultsPlot.load_image(self.image_path)
            except IOError:
                print("Could not read Image: ",path)

    def on_load_trace(self, evt):
        wc = "CSV Files (*.csv)|*.csv|TXT Files (*.txt)|*.txt"
        wc += "|All Files|*.*;*"
        with wx.FileDialog(self, "Load Trace", wildcard=wc, \
                           style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return
            pathname = fd.GetPath()
            self.oNB.TraceTab.trace_clear()
            try:
                with open(pathname, "r") as f:
                    reader = csv.reader(f,delimiter=",", \
                                        quotechar="#")
                    for row in reader:
                        try:
                            x, y = int(row[0]), int(row[1])
                            self.oNB.TraceTab.trace_add(x,y)
                        except ValueError:
                            continue
                self.plot_trace()
            except IOError:
                msg = "Could not write to file: "+pathname
                raise IOError(msg)

    def on_save_trace(self, evt):
        wc = "CSV Files (*.csv)|*.csv|TXT Files (*.txt)|*.txt"
        wc += "|All Files|*.*;*"
        with wx.FileDialog(self, "Save Trace", wildcard=wc, \
                           style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return
            pathname = fd.GetPath()
            try:
                with open(pathname, "w") as f:
                    x, y = self.oNB.TraceTab.trace_read()
                    writer = csv.writer(f, delimiter=",")
                    for pos in zip(x,y):
                        writer.writerow(pos)
            except IOError:
                msg = "Could not write to file: "+pathname
                raise IOError(msg)

    def on_exit(self, evt):
        self.Close()


    def demo_setup(self):
        self.image_path = './maps/mensa_bearbeitet.png'
        trace_data = [(640,400),(690,360),(715,350),(740,370),\
                      (765,430),(775,495),(940,500),(980,455),\
                      (990,405),(1050,350),(1150,350)]
        for pos in trace_data:
            self.oNB.TraceTab.trace_add(pos[0], pos[1])
        self.pNB.TracePlot.load_image(self.image_path)
        self.pNB.ResultsPlot.load_image(self.image_path)
        self.plot_trace()

    def empty_setup(self):
        self.image_path = './maps/empty_1200x1200.png'
        self.pNB.TracePlot.load_image(self.image_path)
        self.pNB.ResultsPlot.load_image(self.image_path)


    def get_export_parameters(self):
        wc = "All Files|*.*;*"
        with wx.FileDialog(self, "Save Plot", wildcard=wc, \
                           style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT) as fd:
            if fd.ShowModal() == wx.ID_CANCEL:
                return
            export_path = fd.GetPath()

        config = configparser.ConfigParser()
        config.read(self.config_path)
        new_size = config["Export"]["Size"].replace("(", "").replace(")","")
        dpi = int(config["Export"]["dpi"])
        x, y = [float(el)/2.54 for el in new_size.split(",")]
        return export_path, x, y, dpi

    def on_export_trace_plot(self, evt):
        old_size = self.pNB.TracePlot.fig.get_size_inches()
        export_path, x, y, dpi = self.get_export_parameters()
        self.pNB.TracePlot.fig.set_size_inches(x, y)
        self.pNB.TracePlot.fig.savefig(export_path, dpi=dpi)
        self.pNB.TracePlot.fig.set_size_inches(old_size)
        self.pNB.TracePlot.fig.canvas.draw()

    def on_export_results_plot(self, evt):
        old_size = self.pNB.ResultsPlot.fig.get_size_inches()
        export_path, x, y, dpi = self.get_export_parameters()
        self.pNB.ResultsPlot.fig.set_size_inches(x, y)
        self.pNB.ResultsPlot.fig.savefig(export_path, dpi=dpi)
        self.pNB.ResultsPlot.fig.set_size_inches(old_size)
        self.pNB.ResultsPlot.fig.canvas.draw()

    def on_settings(self, evt):
        SF = SettingsFrame(None)


def generate_walk(config, batch=False):
    # VALIDATION of CONFIG
    simCount = 0
    x = config["x"][:] #[:] necessary for copying instead of just
    y = config["y"][:] #referencing the lists
    worldMap = image2array(config["path"])
    if not validate_trace(x,y,worldMap, batch):
        print("Invalid Initial Trace")
        return
    while simCount <= 100:
        xn, yn = add_simple_noise(x, y, config["pre_noise"])
        if config["method"] == "Interpolation":
            xs, ys = interpolated_walk(xn, yn, factor=config["factor"],\
                                       kind=config["kind"])
        elif config["method"] == "Simulation":
            JohnCleese = SimulatedWalker(xn, yn, config)
            xs, ys = JohnCleese.run_simulation()
        else:
            print("Invalid Method-Option: ",config["Method"])
            return
        if validate_trace(xs, ys, worldMap, batch):
            # return xs, ys, worldMap #nur zum Test
            xfinal, yfinal = add_simple_noise(xs, ys, config["post_noise"])
            pos = np.array([xfinal, yfinal]).T

            vx = (xfinal - np.roll(xfinal,1))[1:]
            vy = (yfinal - np.roll(yfinal,1))[1:]
            vel = np.array([vx, vy]).T

            ax = (vx - np.roll(vx,1))[1:]
            ay = (vy - np.roll(vy,1))[1:]
            acc = np.array([ax, ay]).T
            return pos, vel, acc

        simCount += 1
    print("Could not create a valid walk...I TRIED!!")
    return

def batch_walk(path):
    simulations = read_config_file(path)
    count = 0
    out_path = os.path.split(path)[-1].split(".")[0]+".hdf5"
    with h5py.File(out_path, "w") as f:
        for config in simulations:
            for n in range(config["nr_runs"]):
                label = config["Goal"]
                if label not in f.keys():
                    new_grp = f.create_group(label)
                    new_grp.attrs["Type"] = "Trajectory"
                grp = f[label].create_group(str(count+1))
                pos, vel, acc = generate_walk(config, batch=True)
                grp.create_dataset("Positions", data=pos)
                grp.create_dataset("Velocity", data=vel)
                grp.create_dataset("Acceleration", data=acc)
                for key in [key for key in config if key != "nr_runs"]:
                    grp.attrs[key] = config[key]
                count += 1
        grp_im = f.create_group("Images")
        grp_im.attrs["Type"] = "Images"
        im = cv2.imread(simulations[0]["path"])
        grp_im.create_dataset("OriginalFrame", data=im)
        grp_im.create_dataset("OriginalGoals", data=im)
        comment = ""
        for i, config_dict in enumerate(simulations):
            comment += "Simulation Nr. "+str(i)+":\n"
            for key in config_dict:
                comment += str(key)+":"+str(config_dict[key])+", "
            comment = comment[:-2]+"\n"
        f.attrs["Comment"] = comment
    print("Done...")

def read_config_file(path):
    try:
        simulations = []
        with open(path, "r") as f:
            for line in f.readlines():
                if line and not line.startswith("#"):
                    p = [el.strip() for el in line.strip().split(",")]
                    config = {}
                    config["Goal"] = p[0]
                    config["Origin"] = p[1]
                    config["nr_runs"] = int(p[2])
                    config["method"] = p[3]
                    config["kind"] = p[4]
                    config["factor"] = int(p[5])
                    config["pre_noise"] = float(p[6])
                    config["post_noise"] = float(p[7])
                    config["path"] = p[8]
                    config["x"] = np.fromstring(p[9][1:-1], sep=" ")
                    config["y"] = np.fromstring(p[10][1:-1], sep=" ")
                    assert len(config["x"]) == len(config["y"])
                    simulations.append(config)
        return simulations
    except FileNotFoundError:
        print("Input-File not found...")


if __name__ == "__main__":
    AP = ArgumentParser()
    AP.add_argument('-d', '--demo', action='store_true')
    AP.add_argument('-b', '--batch')
    args = AP.parse_args()
    if args.batch:
        if args.batch[-6:] == ".batch":
            batch_walk(args.batch)
    else:
        app = wx.App()
        SWG = TrajectoryGeneratorGui()
        if args.demo:
            SWG.demo_setup()
        else:
            SWG.empty_setup()
        app.MainLoop()
