import wx
import wx.lib.mixins.listctrl as  listmix

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class EditableListCtrl(wx.ListCtrl, listmix.TextEditMixin):
    def __init__(self, parent, ID=wx.ID_ANY, pos=wx.DefaultPosition, \
                 size=wx.DefaultSize, style=0):
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
        listmix.TextEditMixin.__init__(self)


class OptionsTraceTab(wx.Panel):
    def __init__(self, parent):
        self.index = 0
        wx.Panel.__init__(self, parent, id=wx.ID_ANY)
        outer_box = wx.BoxSizer(wx.VERTICAL)
        sizer = wx.BoxSizer(wx.VERTICAL)
        btn_top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)

        """ Definitions """
        self.btn_record = wx.Button(self, wx.ID_ANY, \
                                    label="Record Trace")
        self.btn_clear = wx.Button(self, wx.ID_ANY, label="Clear Trace")
        btn_top_sizer.Add(self.btn_record,  wx.ALL, 5)
        btn_top_sizer.Add(self.btn_clear,  wx.ALL, 5)

        self.trace = EditableListCtrl(self, wx.ID_ANY, \
                                        style=wx.LC_REPORT|\
                                        wx.LC_EDIT_LABELS)
        self.trace.InsertColumn(0, "#", width=30)
        self.trace.InsertColumn(1, "x")
        self.trace.InsertColumn(2, "y")

        self.btn_add = wx.Button(self, size=(30,30), label="+")
        self.btn_rem = wx.Button(self, size=(30,30), label="-")
        self.btn_load = wx.Button(self, label="Load")
        self.btn_save = wx.Button(self, label="Save")
        btn_bottom_sizer.Add(self.btn_add, wx.ALL, 5)
        btn_bottom_sizer.Add(self.btn_rem, wx.ALL, 5)
        btn_bottom_sizer.Add(self.btn_load, wx.ALL, 5)
        btn_bottom_sizer.Add(self.btn_save, wx.ALL, 5)

        sizer.Add(btn_top_sizer, flag=wx.ALL, border=5, proportion=.1)
        sizer.Add(self.trace, proportion=8, \
                  flag=wx.EXPAND|wx.ALL, border=5)
        sizer.Add(btn_bottom_sizer, flag=wx.ALL, border=5, proportion=.1)
        outer_box.Add(sizer, 1, wx.ALL, border=10)
        self.SetSizer(outer_box)


    def trace_add(self, x, y):
        self.trace.InsertStringItem(self.index,  str(self.index))
        self.trace.SetStringItem(self.index, 1, str(x))
        self.trace.SetStringItem(self.index, 2, str(y))
        self.index += 1

    def trace_read(self):
        valid_x, valid_y = self.trace_validate()
        if False in (valid_x, valid_y):
            return valid_x, valid_y

        trace_x, trace_y = [], []
        for i in range(self.index):
            # x = int(self.trace.GetItemText(i, 1))
            # y = int(self.trace.GetItemText(i, 2))
            trace_x.append(int(self.trace.GetItemText(i, 1)))
            trace_y.append(int(self.trace.GetItemText(i, 2)))
        return trace_x, trace_y

    def trace_clear(self):
        self.trace.DeleteAllItems()
        self.index = 0


    def trace_remove_selected(self):
        selection = []
        idx = self.trace.GetNextSelected(-1)
        while idx != -1:
            selection.append(idx)
            idx = self.trace.GetNextSelected(idx)
        for el in selection[::-1]:
            self.trace.DeleteItem(el)
            self.index -= 1
        idx = 0
        count = 0
        while idx != -1:
            self.trace.SetItemText(idx, str(count))
            idx = self.trace.GetNextItem(idx)
            count += 1

    def trace_validate(self):
        valid_x, valid_y = True, True
        if self.index == 0:
            return False, False
        for i in range(self.index):
            try:
                int(self.trace.GetItemText(i, 1))
            except ValueError:
                valid_x = False
            try:
                int(self.trace.GetItemText(i, 2))
            except ValueError:
                valid_y = False
        return valid_x, valid_y


if __name__ == "__main__":
    class DemoFrame(wx.Frame):
        def __init__(self):
            wx.Frame.__init__(self, None, wx.ID_ANY,
                           "Notebook Test"
                            )
                            #,size=(330,700))
            panel = wx.Panel(self)
            self.notebook = wx.Notebook(panel)
            self.simTab = OptionsTraceTab(self.notebook)
            self.notebook.AddPage(self.simTab, "Simulation")
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
