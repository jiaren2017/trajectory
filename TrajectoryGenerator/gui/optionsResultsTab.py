import wx

class OptionsResultsTab(wx.Panel):
    def __init__(self, parent):
        self.index = 0
        wx.Panel.__init__(self, parent, id=wx.ID_ANY)
        outer_box = wx.BoxSizer(wx.VERTICAL)
        sizer = wx.BoxSizer(wx.VERTICAL)
        btn_top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)

        """ Definitions """
        self.btn_select_all = wx.Button(self, wx.ID_ANY, \
                                        label="Select All")
        self.btn_export = wx.Button(self, wx.ID_ANY, \
                                    label="Export Walks")
        self.btn_clear = wx.Button(self, wx.ID_ANY, label="Clear Walks")
        btn_top_sizer.Add(self.btn_select_all, wx.ALL, 5)
        btn_top_sizer.Add(self.btn_export,  wx.ALL, 5)
        btn_top_sizer.Add(self.btn_clear,  wx.ALL, 5)

        self.walks = wx.ListCtrl(self, wx.ID_ANY, \
                                        style=wx.LC_REPORT|wx.LC_VRULES)
        self.walks.InsertColumn(0, "#", format=wx.LIST_FORMAT_LEFT, \
                                width=30)
        self.walks.InsertColumn(1, "Label", format=wx.LIST_FORMAT_LEFT, \
                                width=200)

        self.btn_rem = wx.Button(self, size=(30,30), label="-")
        btn_bottom_sizer.Add(self.btn_rem, wx.ALL, 5)

        sizer.Add(btn_top_sizer, flag=wx.ALL, border=5, proportion=.1)
        sizer.Add(self.walks, proportion=8, \
                  flag=wx.EXPAND|wx.ALL, border=5)
        sizer.Add(btn_bottom_sizer, flag=wx.ALL, border=5, proportion=.1)
        outer_box.Add(sizer, 1, wx.ALL, border=10)
        self.SetSizer(outer_box)


    def add_walk(self, label):
        self.walks.InsertItem(self.index,  str(self.index))
        self.walks.SetItem(self.index, 1, label)
        self.index += 1
        return self.index-1

    def walk_read(self, idx):
        return self.walks.GetItemText(idx, 1)

    def walk_clear(self):
        self.walks.DeleteAllItems()
        self.index = 0

    def walk_get_selected(self):
        selection = []
        idx = self.walks.GetNextSelected(-1)
        while idx != -1:
            selection.append(idx)
            idx = self.walks.GetNextSelected(idx)
        return selection

    def walk_remove_selected(self, selection):
        for el in selection[::-1]:
            self.walks.DeleteItem(el)
            self.index -= 1
        idx = self.walks.GetNextItem(-1)
        count = 0
        while idx != -1:
            self.walks.SetItemText(idx, str(count))
            idx = self.walks.GetNextItem(idx)
            count += 1

    def walk_select_all(self):
        idx = -1
        while 1:
            idx = self.walks.GetNextItem(idx,
                                wx.LIST_NEXT_ALL,
                                wx.LIST_STATE_DONTCARE)
            if idx == -1:
                return
            else:
                self.walks.Select(idx, on=1)


if __name__ == "__main__":
    class DemoFrame(wx.Frame):
        def __init__(self):
            wx.Frame.__init__(self, None, wx.ID_ANY,
                           "Notebook Test"
                            # )
                            ,size=(330,700))
            self.panel = OptionsResultsTab(self)
            self.Bind(wx.EVT_BUTTON, self.onTest, \
                      self.panel.btn_export)
            self.Bind(wx.EVT_BUTTON, self.onClear, \
                      self.panel.btn_clear)
            self.Bind(wx.EVT_BUTTON, self.onRem, \
                      self.panel.btn_rem)
            self.Centre()
            self.Show()

        def onClear(self,evt):
            self.panel.walk_clear()
        def onRem(self, evt):
            sel = self.panel.walk_get_selected()
            self.panel.walk_remove_selected(sel)

        def onTest(self, evt):
            labels = ["Yo Mama", "so FAT!"]
            for l in labels:
                self.panel.add_walk(l)

        def onExit(self, evt):
            self.Destroy()

        def onSize(self, evt):
            size = self.GetSize()
            print(size)
    app = wx.App()
    frame = DemoFrame()
    app.MainLoop()
