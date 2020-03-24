'''
Controller class for the differnetial evolution class diffev
Takes care of stopping and starting - output to the gui as well
as some input from dialog boxes.
Programmer Matts Bjorck
Last Changed 2009 05 12
'''
try:
    from . import locate_path
except:
    import locate_path
import os, sys
script_path = locate_path.module_path_locator()
DaFy_path = os.path.dirname(os.path.dirname(script_path))
sys.path.append(DaFy_path)
sys.path.append(os.path.join(DaFy_path,'EnginePool'))
sys.path.append(os.path.join(DaFy_path,'util'))
# import wx, traceback
from io import StringIO
# import  wx.lib.newevent
from wx.lib.masked import NumCtrl

import diffev, fom_funcs
import filehandling as io
import numpy as np
from PyQt5.QtCore import QThread
from PyQt5 import QtCore

#==============================================================================
class SolverController(QtCore.QObject):
    '''
    Class to take care of the GUI - solver interaction.
    Implements dialogboxes for setting parameters and controls
    for the solver routine. This is where the application specific
    code are used i.e. interfacing the optimization code to the GUI.
    '''

    def __init__(self, model, config = None):
        # Create the optimizer we are using. In this case the standard
        # Differential evolution optimizer.
        self.optimizer = diffev.DiffEv()
        # Store the parent we need this to bind the different components to
        # the optimization algorithm.
        self.model = model
        self.config = config

        # Just storage of the starting values for the paramters before
        # the fit is started so the user can go back to the start values
        self.start_parameter_values = None
        # The level used for error bar calculations
        self.fom_error_bars_level = 1.05

        # Setup the output functions.
        self.optimizer.set_text_output_func(self.TextOutput)
        self.optimizer.set_plot_output_func(self.PlotOutput)
        self.optimizer.set_parameter_output_func(self.ParameterOutput)
        self.optimizer.set_fitting_ended_func(self.FittingEnded)
        self.optimizer.set_autosave_func(self.AutoSave)

        # self.parent.Bind(EVT_FITTING_ENDED, self.OnFittingEnded)

        # Now load the default configuration
        self.ReadConfig()


    def ReadConfig(self):
        '''ReadConfig(self) --> None

        Reads the parameter that should be read from the config file.
        And set the parameters in both the optimizer and this class.
        '''
        # Define all the options we want to set
        options_float = ['km', 'kr', 'pop mult', 'pop size',\
                         'max generations', 'max generation mult',\
                         'sleep time','max log elements','errorbar level',
                         'autosave interval', 'parallel processes',
                         'parallel chunksize', 'allowed fom discrepancy']
        setfunctions_float = [self.optimizer.set_km, self.optimizer.set_kr,
                              self.optimizer.set_pop_mult,
                              self.optimizer.set_pop_size,
                              self.optimizer.set_max_generations,
                              self.optimizer.set_max_generation_mult,
                              self.optimizer.set_sleep_time,
                              self.optimizer.set_max_log,
                              self.set_error_bars_level,
                              self.optimizer.set_autosave_interval,
                              self.optimizer.set_processes,
                              self.optimizer.set_chunksize,
                              self.optimizer.set_fom_allowed_dis,
                              ]

        options_bool = ['use pop mult', 'use max generations',
                        'use start guess', 'use boundaries',
                         'use parallel processing', 'use autosave',
                         'save all evals']
        setfunctions_bool = [ self.optimizer.set_use_pop_mult,
                              self.optimizer.set_use_max_generations,
                              self.optimizer.set_use_start_guess,
                              self.optimizer.set_use_boundaries,
                              self.optimizer.set_use_parallel_processing,
                              self.optimizer.set_use_autosave,
                              self.set_save_all_evals,
                              ]

        # Make sure that the config is set
        if self.config:
            # Start witht the float values
            for index in range(len(options_float)):
                try:
                    val = self.config.get_float('solver', options_float[index])
                except io.OptionError as e:
                    print('Could not locate option solver.' +\
                            options_float[index])
                else:
                    setfunctions_float[index](val)

            # Then the bool flags
            for index in range(len(options_bool)):
                try:
                    val = self.config.get_boolean('solver',\
                            options_bool[index])
                except io.OptionError as e:
                    print('Could not read option solver.' +\
                            options_bool[index])
                else:
                    setfunctions_bool[index](val)
            try:
                val = self.config.get('solver', 'create trial')
            except io.OptionError as e:
                print('Could not read option solver.create trial')
            else:
                try:
                    self.optimizer.set_create_trial(val)
                except LookupError:
                    print('The mutation scheme %s does not exist'%val)

    def WriteConfig(self):
        ''' WriteConfig(self) --> None

        Writes the current configuration of the solver to file.
        '''

        # Define all the options we want to set
        options_float = ['km', 'kr', 'pop mult', 'pop size',\
                         'max generations', 'max generation mult',\
                         'sleep time', 'max log elements','errorbar level',\
                         'autosave interval',\
                        'parallel processes', 'parallel chunksize',
                         'allowed fom discrepancy']
        set_float = [self.optimizer.km, self.optimizer.kr,
                          self.optimizer.pop_mult,\
                          self.optimizer.pop_size,\
                         self.optimizer.max_generations,\
                         self.optimizer.max_generation_mult,\
                         self.optimizer.sleep_time,\
                        self.optimizer.max_log, \
                        self.fom_error_bars_level,\
                         self.optimizer.autosave_interval,\
                        self.optimizer.processes,\
                        self.optimizer.chunksize,\
                      self.optimizer.fom_allowed_dis
                     ]

        options_bool = ['use pop mult', 'use max generations',
                        'use start guess', 'use boundaries',
                        'use parallel processing', 'use autosave',
                        'save all evals',
                        ]
        set_bool = [ self.optimizer.use_pop_mult,
                     self.optimizer.use_max_generations,
                     self.optimizer.use_start_guess,
                     self.optimizer.use_boundaries,
                     self.optimizer.use_parallel_processing,
                     self.optimizer.use_autosave,
                     self.save_all_evals,
                     ]

        # Make sure that the config is set
        if self.config:
            # Start witht the float values
            for index in range(len(options_float)):
                try:
                    val = self.config.set('solver', options_float[index],\
                            set_float[index])
                except io.OptionError as e:
                    print('Could not locate save solver.' +\
                            options_float[index])

            # Then the bool flags
            for index in range(len(options_bool)):
                try:
                    val = self.config.set('solver',\
                            options_bool[index], set_bool[index])
                except io.OptionError as e:
                    print('Could not write option solver.' +\
                            options_bool[index])

            try:
                self.config.set('solver', 'create trial',\
                    self.optimizer.get_create_trial())
            except io.OptionError as e:
                print('Could not write option solver.create trial')

    def ParametersDialog(self, frame):
        '''ParametersDialog(self, frame) --> None

        Shows the Parameters dialog box to set the parameters for the solver.
        '''
        # Update the configuration if a model has been loaded after
        # the object have been created..
        self.ReadConfig()
        fom_func_name = self.model.fom_func.__name__
        if not fom_func_name in fom_funcs.func_names:
            ShowWarningDialog(self.parent, 'The loaded fom function, '\
            + fom_func_name+ ', does not exist ' + \
            'in the local fom_funcs file. The fom fucntion has been' +
            ' temporary added to the list of availabe fom functions')
            fom_funcs.func_names.append(fom_func_name)
            exectext = 'fom_funcs.' + fom_func_name +\
                        ' = self.parent.model.fom_func'
            exec(exectext in locals(), globals())

        dlg = SettingsDialog(frame, self.optimizer, self,fom_func_name)

        def applyfunc(object):
            self.WriteConfig()
            self.parent.model.set_fom_func(\
                    eval('fom_funcs.'+object.get_fom_string()))

        dlg.set_apply_change_func(applyfunc)

        dlg.ShowModal()
        #if dlg.ShowModal() == wx.ID_OK:
        #    pass
        dlg.Destroy()

    def TextOutput(self, text):
        '''TextOutput(self, text) --> None
        Function to present the output from the optimizer to the user.
        Takes a string as input.
        '''
        #self.parent.main_frame_statusbar.SetStatusText(text, 0)
        pass
        #evt = update_text(text = text)
        #wx.PostEvent(self.parent, evt)

    def PlotOutput(self, solver):
        ''' PlotOutput(self, solver) --> None
        Solver to present the graphical output from the optimizer to the
        user. Takes the solver as input argument and picks out the
        variables to show in the GUI.
        
        #print 'sending event plotting'
        #_post_solver_event(self.parent, solver, desc = 'Fitting update')
        evt = update_plot(model = solver.get_model(), \
                fom_log = solver.get_fom_log(), update_fit = solver.new_best,\
                desc = 'Fitting update')
        wx.PostEvent(self.parent, evt)
        # Hard code the events for the plugins so that they can be run syncrously.
        # This is important since the Refelctevity model, for example, relies on the
        # current state of the model.
        try:
            self.parent.plugin_control.OnFittingUpdate(evt)
            #pass
        except Exception as e:
            print('Error in plot output:\n' + repr(e))
        '''
        pass
        # self.parent.update_plot_data_view_upon_simulation()
        

    def ParameterOutput(self, solver):
        '''ParameterOutput(self, solver) --> none

        Function to send an update event to update windows that displays
        the parameters to update the values.
        Takes the solver as input argument and picks out the variables to
        show in the GUI.
        
        evt = update_parameters(values = solver.best_vec.copy(),\
                new_best = solver.new_best,\
                population = solver.pop_vec,\
                max_val = solver.par_max, \
                min_val = solver.par_min, \
                fitting = True,\
                desc = 'Parameter Update', update_errors = False,\
                permanent_change = False)
        wx.PostEvent(self.parent, evt)
       ''' 
        #print(len(solver.best_vec.copy()))
        best_vec = solver.best_vec.copy()
        offset = 0
        index_for_update = [i for i in range(len(solver.model.parameters.data)) if solver.model.parameters.data[i][2] in [True,'True',1]]
        for i in range(len(index_for_update)):
            solver.model.parameters.data[index_for_update[i]][1] = best_vec[i]

        #self.parent.update_par()
    
    def ModelLoaded(self):
        '''ModelLoaded(self) --> None

        Function that takes care of resetting everything when a model has
        been loaded.
        

        evt = update_plot(model = self.optimizer.get_model(), \
                fom_log = self.optimizer.get_fom_log(), update_fit = False,\
                desc = 'Model loaded')
        wx.PostEvent(self.parent, evt)

        # Update the parameter plot ...
        if self.optimizer.setup_ok:
            # remeber to add a check
            solver = self.optimizer
            try:
                evt = update_parameters(values = solver.best_vec.copy(),\
                    new_best = False,\
                    population = solver.pop_vec,\
                    max_val = solver.par_max, \
                    min_val = solver.par_min, \
                    fitting = True,\
                    desc = 'Parameter Update', update_errors = False,\
                    permanent_change = False)
            except:
                print('Could not create data for paraemters')
            else:
                wx.PostEvent(self.parent, evt)
        '''
        pass

    def AutoSave(self):
        '''DoAutoSave(self) --> None

        Function that conducts an autosave of the model.
        '''
        io.save_gx(self.model.get_filename(), self.model, \
                self.optimizer, self.config)
        #print 'AutoSaved!'

    def FittingEnded(self, solver):
        '''FittingEnded(self, solver) --> None

        function used to post an event when the fitting has ended.
        This must be done since it is not htread safe otherwise. Same GUI in
        two threads when dialogs are run. dangerous...
        
        evt = fitting_ended(solver = solver, desc = 'Fitting Ended')
        wx.PostEvent(self.parent, evt)
        '''
        pass



    def OnFittingEnded(self, evt):
        '''OnFittingEnded(self, solver) --> None

        Callback when fitting has ended. Takes care of cleaning up after
        the fit. Calculates errors on the parameters and updates the grid.
        
        solver = evt.solver
        if solver.error:
            ShowErrorDialog(self.parent, solver.error)
            return

        message = 'Do you want to keep the parameter values from' +\
                'the fit?'
        dlg = wx.MessageDialog(self.parent, message,'Keep the fit?',
            wx.YES_NO|wx.ICON_QUESTION)
        if dlg.ShowModal() == wx.ID_YES:
            evt = update_parameters(values = solver.best_vec.copy(),\
                desc = 'Parameter Update', new_best = True, \
                update_errors = False, fitting = False,\
                 permanent_change = True)
            wx.PostEvent(self.parent, evt)
        else:
            #print 'Resetting the values in the grid to ',\
            #    self.start_parameter_values
            evt = update_parameters(values = solver.start_guess,\
                desc = 'Parameter Update', new_best = True, \
                update_errors = False, fitting = False,\
                 permanent_change = False)
            wx.PostEvent(self.parent, evt)
        '''
        pass

    def CalcErrorBars(self):
        '''CalcErrorBars(self) -- None

        Method that calculates the errorbars for the fit that has been
        done. Note that the fit has to been conducted before this is runned.
        '''
        if self.optimizer.start_guess != None and not self.optimizer.running:
            n_elements = len(self.optimizer.start_guess)
            #print 'Number of elemets to calc errobars for ', n_elements
            error_values = []
            for index in range(n_elements):
                # calculate the error
                # TODO: Check the error bar buisness again and how to treat
                # Chi2
                #print "senor",self.fom_error_bars_level
                try:
                    (error_low, error_high) = self.optimizer.calc_error_bar(\
                                            index, self.fom_error_bars_level)
                except:
                    raise diffev.ErrorBarError
                error_str = '(%.3e, %.3e)'%(error_low, error_high)
                error_values.append(error_str)
            return error_values
        else:
            raise ErrorBarError()

    def ProjectEvals(self, parameter):
        ''' ProjectEvals(self, parameter) --> prameter, fomvals

        Projects the parameter number parameter on one axis and returns
        the fomvals.
        '''
        model  = self.model
        row = model.parameters.get_pos_from_row(parameter)
        if self.optimizer.start_guess != None and not self.optimizer.running:
            return self.optimizer.par_evals[:,row],\
                self.optimizer.fom_evals[:]
        else:
            raise ErrorBarError()

    def ScanParameter(self, parameter, points):
        '''ScanParameter(self, parameter, points)
            --> par_vals, fom_vals

        Scans one parameter and records its fom value as a function
        of the parameter value.
        
        row = parameter
        model = self.parent.model
        (funcs, vals) = model.get_sim_pars()
        minval = model.parameters.get_data()[row][3]
        maxval = model.parameters.get_data()[row][4]
        parfunc = funcs[model.parameters.get_sim_pos_from_row(row)]
        step = (maxval - minval)/points
        par_vals = np.arange(minval, maxval + step, step)
        fom_vals = np.array([])

        par_name = model.parameters.get_data()[row][0]
        dlg = wx.ProgressDialog("Scan Parameter",
                               "Scanning parameter " + par_name,
                               maximum = len(par_vals),
                               parent=self.parent,
                               style = wx.PD_APP_MODAL| wx.PD_ELAPSED_TIME
                               | wx.PD_REMAINING_TIME | wx.PD_AUTO_HIDE)
        try:
            # Start with setting all values
            [f(v) for (f, v) in zip(funcs, vals)]
            for par_val in par_vals:
                parfunc(par_val)
                fom_vals = np.append(fom_vals, model.evaluate_fit_func())
                dlg.Update(len(fom_vals))
        except Exception as e:
            dlg.Destroy()
            outp = StringIO.StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            ShowWarningDialog(self.parent, 'Error while evaluatating the' + \
            ' simulation and fom. Please check so it is possible to simulate'+\
            ' your model. Detailed output below: \n\n' + val)
        else:
            dlg.Destroy()
        # resetting the scanned parameter
        funcs[row](vals[row])
        return par_vals, fom_vals
        '''
        pass

    def ResetOptimizer(self):
        '''ResetOptimizer(self) --> None

        Resets the optimizer - clears the memory and special flags.
        '''
        self.start_parameter_values = None


    def StartFit(self, signal = None):
        ''' StartFit(self) --> None
        Function to start running the fit
        '''
        # Make sure that the config of the solver is updated..
        self.ReadConfig()
        model = self.model
        # Reset all the errorbars
        model.parameters.clear_error_pars()
        #self.start_parameter_values = model.get_fit_values()
        self.optimizer.model = model
        self.optimizer.start_fit(signal)


    def StartFit_old(self):
        ''' StartFit(self) --> None
        Function to start running the fit
        '''
        # Make sure that the config of the solver is updated..
        
        self.ReadConfig()
        model = self.model
        # Reset all the errorbars
        self.parent.new_thread = QThread()

        model.parameters.clear_error_pars()

        #self.start_parameter_values = model.get_fit_values()
        self.optimizer.model = model
        self.parent.new_thread.started.connect(self.optimizer.start_fit)
        # self.parent.update_plot_data_view_upon_simulation.moveToThread(self.parent.new_thread)
        self.parent.new_thread.start()
        #print 'Optimizer starting'

    def StopFit(self):
        ''' StopFit(self) --> None
        Function to stop a running fit
        '''
        self.optimizer.stop_fit()

    def ResumeFit(self):
        ''' ResumeFit(self) --> None

        Function to resume the fitting after it has been stopped
        '''
        # Make sure the settings are updated..
        self.ReadConfig()
        model = self.model
        # Remove all previous erros ...

        self.optimizer.resume_fit(model)

    def IsFitted(self):
        '''IsFitted(self) --> bool

        Returns true if a fit has been started otherwise False
        '''
        return self.optimizer.start_guess != None


    def set_error_bars_level(self, value):
        '''set_error_bars_level(value) --> None

        Sets the value of increase of the fom used for errorbar calculations
        '''
        if value < 1:
            raise ValueError('fom_error_bars_level has to be above 1')
        else:
            self.fom_error_bars_level = value

    def set_save_all_evals(self, value):
        '''Sets the boolean value to save all evals to file
        '''
        self.save_all_evals = bool(value)

class GenericError(Exception):
    ''' Just a empty class used for inheritance. Only useful
    to check if the errors are originating from the model library.
    All these errors are controllable. If they not originate from
    this class something has passed trough and that should be impossible '''
    pass

class ErrorBarError(GenericError):
    '''Error class for the fom evaluation'''
    def __init__(self):
        ''' __init__(self) --> None'''
        #self.error_message = error_message

    def __str__(self):
        text = 'Could not evaluate the error bars. A fit has to be run ' +\
                'before they can be calculated'
        return text


