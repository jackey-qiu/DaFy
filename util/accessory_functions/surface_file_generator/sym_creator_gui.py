import wx
import sys
sys.path.append('D:\\Google Drive\\useful codes\\symmetry-creator')
import symmetry_creator_test4 as sym
import numpy as num
#I add line1
#I add line2
class sym_creator_gui(wx.Frame):
    def __init__(self,parent,id,title):
        frame=wx.Frame.__init__(self,parent,id,title,size=(480,615))
        p1=wx.Panel(self)
        bt_calculate_id=1
        bt_get_sym_id=2
        bt_print_id=3
        
        self.tc_bulk_a=wx.TextCtrl(p1,-1)
        self.tc_bulk_b=wx.TextCtrl(p1,-1)
        self.tc_bulk_c=wx.TextCtrl(p1,-1)
        self.tc_surf_a=wx.TextCtrl(p1,-1)
        self.tc_surf_b=wx.TextCtrl(p1,-1)
        self.tc_surf_c=wx.TextCtrl(p1,-1)
        
        self.tc_el_1=wx.TextCtrl(p1,-1)
        self.tc_el_2=wx.TextCtrl(p1,-1)
        self.tc_el_1_x=wx.TextCtrl(p1,-1)
        self.tc_el_1_y=wx.TextCtrl(p1,-1)
        self.tc_el_1_z=wx.TextCtrl(p1,-1)
        self.tc_el_2_x=wx.TextCtrl(p1,-1)
        self.tc_el_2_y=wx.TextCtrl(p1,-1)
        self.tc_el_2_z=wx.TextCtrl(p1,-1)
        
        self.tc_a_x=wx.TextCtrl(p1,-1)
        self.tc_a_y=wx.TextCtrl(p1,-1)
        self.tc_a_z=wx.TextCtrl(p1,-1)
        self.tc_b_x=wx.TextCtrl(p1,-1)
        self.tc_b_y=wx.TextCtrl(p1,-1)
        self.tc_b_z=wx.TextCtrl(p1,-1)
        self.tc_c_x=wx.TextCtrl(p1,-1)
        self.tc_c_y=wx.TextCtrl(p1,-1)
        self.tc_c_z=wx.TextCtrl(p1,-1)
        
        self.tc_sym_path=wx.TextCtrl(p1,-1,size=(250,20))
        self.tc_element=wx.TextCtrl(p1,19,size=(50,20))
        self.tc_number=wx.TextCtrl(p1,size=(50,20))
        self.tc_file_head=wx.TextCtrl(p1)
        self.tc_save_path=wx.TextCtrl(p1,size=(240,20))
        
        self.cb1=wx.ComboBox(p1, -1, choices=['True','False'], style=wx.CB_READONLY)
        self.cb_surf_frat=wx.CheckBox(p1,-1)
        self.cb_surf_ans=wx.CheckBox(p1,-1)
        self.cb_bulk_frat=wx.CheckBox(p1,-1)
        self.cb_bulk_ans=wx.CheckBox(p1,-1)

        box_p1=wx.BoxSizer(wx.VERTICAL)
        sb_1=wx.StaticBox(p1,-1,'Crystal Info', (5, 5),size=(450,90))
        wx.StaticBox(p1,-1,'Asymmetry atoms', (5, 100),size=(450,90))
        sb_2=wx.StaticBox(p1,-1,'Bulk-->surface', (5, 195),size=(450,120))
        gs_1=wx.GridSizer(3,4,1,1)
        gs_1.AddMany([(wx.StaticText(p1,-1),1,wx.ALIGN_CENTER_HORIZONTAL|wx.TOP,3),\
                      (wx.StaticText(p1,-1,'a'),1,wx.ALIGN_CENTER_HORIZONTAL|wx.TOP,3),\
                      (wx.StaticText(p1,-1,'b'),1,wx.ALIGN_CENTER_HORIZONTAL|wx.TOP,3),\
                      (wx.StaticText(p1,-1,'c'),1,wx.ALIGN_CENTER_HORIZONTAL|wx.TOP,3),\
                      (wx.StaticText(p1,-1,'bulk'),1,wx.ALIGN_LEFT),\
                      (self.tc_bulk_a,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.tc_bulk_b,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.tc_bulk_c,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (wx.StaticText(p1,-1,'surf'),1,wx.ALIGN_LEFT),\
                      (self.tc_surf_a,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.tc_surf_b,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.tc_surf_c,1,wx.ALIGN_CENTER_HORIZONTAL)])
        box_p1.Add(gs_1,0,wx.ALL,15)
        
        gs_11=wx.GridSizer(3,4,1,1)
        gs_11.AddMany([(wx.StaticText(p1,-1,'Element'),1,wx.ALIGN_CENTER_HORIZONTAL|wx.TOP,3),\
                      (wx.StaticText(p1,-1,'x'),1,wx.ALIGN_CENTER_HORIZONTAL|wx.TOP,3),\
                      (wx.StaticText(p1,-1,'y'),1,wx.ALIGN_CENTER_HORIZONTAL|wx.TOP,3),\
                      (wx.StaticText(p1,-1,'z'),1,wx.ALIGN_CENTER_HORIZONTAL|wx.TOP,3),\
                      (self.tc_el_1,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.tc_el_1_x,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.tc_el_1_y,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.tc_el_1_z,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.tc_el_2,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.tc_el_2_x,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.tc_el_2_y,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.tc_el_2_z,1,wx.ALIGN_CENTER_HORIZONTAL)])
        box_p1.Add(gs_11,0,wx.ALL,15)
        
        gs_2=wx.GridSizer(4,4,1,1)
        gs_2.AddMany([(wx.StaticText(p1,-1),1,wx.ALIGN_CENTER_HORIZONTAL|wx.TOP,3),\
                      (wx.StaticText(p1,-1,'x'),1,wx.ALIGN_CENTER_HORIZONTAL|wx.TOP,3),\
                      (wx.StaticText(p1,-1,'y'),1,wx.ALIGN_CENTER_HORIZONTAL|wx.TOP,3),\
                      (wx.StaticText(p1,-1,'z'),1,wx.ALIGN_CENTER_HORIZONTAL|wx.TOP,3),\
                      (wx.StaticText(p1,-1,'a'),1,wx.ALIGN_LEFT),\
                      (self.tc_a_x,1,wx.ALIGN_LEFT),\
                      (self.tc_a_y,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.tc_a_z,1,wx.ALIGN_CENTER_HORIZONTAL),\
                      (wx.StaticText(p1,-1,'b'),1,wx.ALIGN_LEFT),\
                      (self.tc_b_x,1,wx.ALIGN_LEFT),\
                      (self.tc_b_y,1,wx.ALIGN_LEFT),\
                      (self.tc_b_z,1,wx.ALIGN_LEFT),\
                      (wx.StaticText(p1,-1,'c'),1,wx.ALIGN_LEFT),\
                      (self.tc_c_x,1,wx.ALIGN_LEFT),\
                      (self.tc_c_y,1,wx.ALIGN_LEFT),\
                      (self.tc_c_z,1,wx.ALIGN_LEFT)])
        box_p1.Add(gs_2,0,wx.ALL,15)
        box_p1.Add((-1,15))
        
        box_h1=wx.BoxSizer(wx.HORIZONTAL)
        st_1=wx.StaticText(p1,-1,'symmetry file path')
        bt1=wx.Button(p1, bt_calculate_id, 'calculate')
        box_h1.Add(st_1,0,wx.RIGHT,10)
        box_h1.Add(self.tc_sym_path,0,wx.RIGHT,10)
        box_h1.Add(bt1)
        box_p1.Add(box_h1,0,wx.ALL,15)
        
        wx.StaticLine(p1, -1, (5, 365), (450,2))
        
        bt2=wx.Button(p1, bt_get_sym_id, 'Get Symmetry')
        box_h2=wx.BoxSizer(wx.HORIZONTAL)
        box_h2.Add(wx.StaticText(p1,-1,'Element'),0,wx.RIGHT,2)
        box_h2.Add(self.tc_element,0,wx.RIGHT,6)
        box_h2.Add(wx.StaticText(p1,-1,'Number'),0,wx.RIGHT,2)
        box_h2.Add(self.tc_number,0,wx.RIGHT,6)
        box_h2.Add(wx.StaticText(p1,-1,'Print symmetry file'),0,wx.RIGHT,2)
        box_h2.Add(self.cb1,0,wx.RIGHT,10)
        box_h2.Add(bt2,0,wx.RIGHT,6)
        box_p1.Add(box_h2,0,wx.ALL,10)
        box_p1.Add((-1,10))
        sb_3=wx.StaticBox(p1,-1,'Create symmetry files', (5, 320),size=(450,100))
        
        sb_4=wx.StaticBox(p1,-1,'Print files', (5, 420),size=(450,150))
        box_h3=wx.BoxSizer(wx.HORIZONTAL)
        box_h3.Add(wx.StaticText(p1,-1,'file head'),0,wx.RIGHT,2)
        box_h3.Add(self.tc_file_head,0,wx.RIGHT,5)
        box_h3.Add(wx.StaticText(p1,-1,'save path'),0,wx.RIGHT,2)
        box_h3.Add(self.tc_save_path,0)
        box_p1.Add(box_h3,0,wx.ALL,10)
        gs_3=wx.GridSizer(3,3,5,40)
        gs_3.AddMany([(wx.StaticText(p1,-1),0,wx.ALIGN_CENTER_HORIZONTAL),\
                      (wx.StaticText(p1,-1,'fract.'),0,wx.ALIGN_CENTER_HORIZONTAL),\
                      (wx.StaticText(p1,-1,'anstr.'),0,wx.ALIGN_CENTER_HORIZONTAL),\
                      (wx.StaticText(p1,-1,'surface'),0,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.cb_surf_frat,0,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.cb_surf_ans,0,wx.ALIGN_CENTER_HORIZONTAL),\
                      (wx.StaticText(p1,-1,'bulk'),0,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.cb_bulk_frat,0,wx.ALIGN_CENTER_HORIZONTAL),\
                      (self.cb_bulk_ans,0,wx.ALIGN_CENTER_HORIZONTAL)])
        box_p1.Add(gs_3,0,wx.ALL,10)
        bt3=wx.Button(p1,bt_print_id,'Print')
        box_p1.Add(bt3,0,wx.ALIGN_RIGHT|wx.RIGHT,30)
        
        p1.SetSizer(box_p1)
        
        self.Bind(wx.EVT_BUTTON, self.calculate, id=bt_calculate_id)
        self.Bind(wx.EVT_BUTTON, self.sym_creator, id=bt_get_sym_id)
        self.Bind(wx.EVT_BUTTON, self.print_sym, id=bt_print_id)
        
        self.Show()
    def calculate(self,event):
        bulk=[float(self.tc_bulk_a.GetValue()),float(self.tc_bulk_b.GetValue()),float(self.tc_bulk_c.GetValue())]
        surf=[float(self.tc_surf_a.GetValue()),float(self.tc_surf_b.GetValue()),float(self.tc_surf_c.GetValue())]
        bulk_to_surf=[[float(self.tc_a_x.GetValue()),float(self.tc_a_y.GetValue()),float(self.tc_a_z.GetValue())],\
                      [float(self.tc_b_x.GetValue()),float(self.tc_b_y.GetValue()),float(self.tc_b_z.GetValue())],\
                      [float(self.tc_c_x.GetValue()),float(self.tc_c_y.GetValue()),float(self.tc_c_z.GetValue())]]
        sym_file=self.tc_sym_path.GetValue()
        asym={self.tc_el_1.GetValue():(float(self.tc_el_1_x.GetValue()),float(self.tc_el_1_y.GetValue()),float(self.tc_el_1_z.GetValue())),\
              self.tc_el_2.GetValue():(float(self.tc_el_2_x.GetValue()),float(self.tc_el_2_y.GetValue()),float(self.tc_el_2_z.GetValue()))}
        self.sym_test=sym.sym_creator(bulk_cell=bulk,surf_cell=surf,bulk_to_surf=bulk_to_surf,asym_atm=asym,sym_file=sym_file)
        self.sym_test.create_bulk_sym()
        self.sym_test.find_atm_bulk()
        self.sym_test.find_atm_surf()
        
    def sym_creator(self,event):
        element=[self.tc_element.GetValue()]
        rn=[float(self.tc_number.GetValue())]
        self.sym_test.set_new_ref_atm_surf(el=element,rn=rn,print_file=bool(self.cb1.GetValue()))
    def print_sym(self,event):
        file_path=self.tc_save_path.GetValue()+self.tc_file_head.GetValue()
        self.sym_test.print_files(filename=file_path,b_f=self.cb_bulk_frat.GetValue(),\
                                  b_a=self.cb_bulk_ans.GetValue(),s_f=self.cb_surf_frat.GetValue(),s_a=self.cb_surf_ans.GetValue())
if __name__=='__main__':
    app=wx.App()
    sym_creator_gui(None,-1,title='symmetry creator')
    app.MainLoop()