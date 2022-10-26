
# Home Credit Python Scoring Library and Workflow
# Copyright © 2017-2020, Pavel Sůva, Marek Teller, Martin Kotek, Jan Zeller,
# Marek Mukenšnabl, Kirill Odintsov, Elena Kuchina, Jan Coufalík, Jan Hynek and
# Home Credit & Finance Bank Limited Liability Company, Moscow, Russia.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time as tm
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd


class InteractionsGUI(tk.Tk):
    """
    GUI part of interactions made in TkInter
    TODO: needs better handling of input data
    TODO: Can this be merged with SelectCategories? (???)

    Args:
        config (config): configuration file
        interactions_result (dict): result from Interactions
        row_var (str): variable from which interaction rows will be created
        column_var (str): variable from which interaction columns will be created
        data (pandas.DataFrame): data from which this class reads
    """

    def __init__(self, config, interactions_result, row_var, column_var, data, *args, **kwargs):
        """
        Initialisation
        """

        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Predictors interaction")
        #  img = tkinter.PhotoImage(file = r'hc_logo.jpg')
        # self.wm_iconbitmap('hc_logo.jpg')

        self.categ = 1
        # controller._def_mtrx has shape of m (rows) x n (columns)
        self._config = config
        self._font = self._config.get("gui", "font")
        self._bad_mtrx = interactions_result['bad_mtrx']
        self._cnt_mtrx = interactions_result['cnt_mtrx']
        self._def_mtrx = interactions_result['def_mtrx']
        self._inter_coord = interactions_result['inter_coord']
        self._row_cat_alias = interactions_result['row_cat_alias']
        self._col_cat_alias = interactions_result['col_cat_alias']
        self._row_bins_numeric = interactions_result['row_bins_numeric']
        self._col_bins_numeric = interactions_result['col_bins_numeric']
        self.m = len(self._def_mtrx.index)
        self.n = len(self._def_mtrx.columns)
        self._row_var = row_var
        self._column_var = column_var
        self.metadata = None
        container = tk.Frame(self)
        container.grid()
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.categories = np.zeros(shape=self._def_mtrx.shape, dtype=int)
        self._data = data
        self.btns = None
        self.frames = {}

        # TODO: please explain this part (???)
        frame = SelectCategories(container, self)  # class instance is created
        # but now, i create mapping of class definition to class instance (???)
        self.frames[SelectCategories] = frame
        frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(SelectCategories)  # and now, I use the class definition to get the frame again
        # and raise it with .tkraise() method (???)

    def show_frame(self, controller):
        """
        (???)

        Args:
            controller ():
        """
        frame = self.frames[controller]
        frame.tkraise()


class SelectCategories(tk.Frame):
    """
    GUI part of Interactions
    TODO: __init__  is long. Should be divided in several subfunctions + unnecessary repetition should be omitted
    TODO: individual parts should be commented, what exactly are initialising
    TODO: explicit variable names, especially for self.subFr<n>
    TODO: If this class cannot be merged with InteractionsGUI, can it be divided in smaller classes?

    Args:
        parent ():
        controller ():
    """

    def __init__(self, parent, controller):
        """
        Initialisation of SelectCategories
        """

        mainFrame = tk.Frame.__init__(self, parent)
        # TODO: rather than numbering, it is far preferable to have explicit names
        # TODO: subFr1 does not tell a lot what exactly is being built using this frame
        # TODO: I guess better name would be self.subframe_titlelab
        self.subFr1 = tk.Frame(mainFrame, bd=5, relief='raised')

        labeltitle = tk.Label(self.subFr1, text='Interaction of {} and {}'.format(
            controller._row_var, controller._column_var), font=(controller._font, 14))
        labeltitle.grid(pady=10, padx=10, row=0, column=0)
        self.subFr1.grid(row=0, column=0, columnspan=3)
        title_ttp = CreateToolTip(labeltitle, "Interaction: xkcd.com/1961/")  # Easter Egg!

        # TODO: rename as self.subframe_rowlab
        self.subFr2 = tk.Frame(mainFrame)
        labelrow = tk.Label(self.subFr2, text=controller._row_var, font=(controller._font, 10))
        labelrow.grid(pady=5, padx=5, row=0, column=0)
        self.subFr2.grid(row=2, column=0, sticky='e')

        # TODO: rename as self.subframe_columnlab
        self.subFr3 = tk.Frame(mainFrame)
        labelcol = tk.Label(self.subFr3, text=controller._column_var, font=(controller._font, 10))
        labelcol.grid(pady=5, padx=5, row=0, column=0)
        self.subFr3.grid(row=1, column=1)

        # TODO: and now, I just don't know what is being created :/ The next code is too complex. (???)
        self.subFr4 = tk.Frame(mainFrame)

        style = ttk.Style()
        style.configure("TButton", width=4, font=(controller._font, 10))

        self.btns = {}
        i = 1
        def_rx_min = controller._def_mtrx.min().min()
        def_rx_span = controller._def_mtrx.max().max() - def_rx_min

        self.categ_colors = [(255, 255, 153),
                             (153, 255, 153),
                             (153, 255, 255),
                             (153, 153, 255),
                             (255, 153, 255),
                             (224, 224, 224),
                             (204, 255, 153),
                             (153, 255, 204),
                             (153, 204, 255),
                             (0, 0, 0)]

        self.stats = {}

        for counter in range(controller.n):
            if (len(controller._col_cat_alias) > 0) & (controller._def_mtrx.columns[counter] != 'null'):
                labelcolbin = tk.Label(
                    self.subFr4, text='{}*'.format(controller._def_mtrx.columns[counter]), font=(controller._font, 8))
                hover_text = '{}:\n{}'.format(controller._def_mtrx.columns[counter], '\n'.join(
                    [key for key in controller._col_cat_alias.keys() if controller._col_cat_alias[key] == controller._def_mtrx.columns[counter]]))
                cat_ttp = CreateToolTip(labelcolbin, hover_text)
            else:
                labelcolbin = tk.Label(
                    self.subFr4, text=controller._def_mtrx.columns[counter], font=(controller._font, 8))
            labelcolbin.grid(pady=10, row=1, column=counter + 2)
        for r in range(controller.m):
            if (len(controller._row_cat_alias) > 0) & (controller._def_mtrx.index[r] != 'null'):
                labelrowbin = tk.Label(
                    self.subFr4, text='{}*'.format(controller._def_mtrx.index[r]), font=(controller._font, 8))
                hover_text = '{}:\n{}'.format(controller._def_mtrx.index[r], '\n'.join(
                    [key for key in controller._row_cat_alias.keys() if controller._row_cat_alias[key] == controller._def_mtrx.index[r]]))
                cat_ttp = CreateToolTip(labelrowbin, hover_text)
            else:
                labelrowbin = tk.Label(self.subFr4, text=controller._def_mtrx.index[r], font=(controller._font, 8))
            labelrowbin.grid(padx=10, row=r + 2, column=1)
            for c in range(controller.n):
                self.btns['btn' + str(i)] = [tk.Button(self.subFr4,
                                                       text=str(round(controller._def_mtrx.iloc[r, c], 4)),
                                                       command=lambda t=i: self.selectCategory(
                                                           'btn' + str(t), controller.categ, controller),
                                                       width=9,
                                                       height=2,
                                                       bg='#%02x%02x%02x' % (round(204 * ((controller._def_mtrx.iloc[r, c] - def_rx_min) / def_rx_span)).astype(int),
                                                                             round(
                                                           204 * (1 - (controller._def_mtrx.iloc[r, c] - def_rx_min) / def_rx_span)).astype(int),
                                                           50),
                                                       # round(0 * (1 - (controller._def_mtrx.iloc[r,c] - def_rx_min) / def_rx_span)).astype(int)),
                                                       fg='white'
                                                       ), 0, r, c, 0]
                self.btns['btn' + str(i)][0].grid(row=r+2, column=c+2)
                i += 1

        self.subFr4.grid(row=2, column=1, sticky='w')

        # TODO: better subframe_name
        self.subFr5 = tk.Frame(mainFrame)

        btn_left = ttk.Button(self.subFr5, text="<-", command=lambda: self.previousCategory(controller), width=6)
        btn_left.grid(row=0, column=0, pady=10)

        btn_right = ttk.Button(self.subFr5, text="->", command=lambda: self.nextCategory(controller), width=6)
        btn_right.grid(row=0, column=2, pady=10)

        labelcateg = tk.Label(self.subFr5, text="Select category {}".format(
            controller.categ), font=(controller._font, 10))
        labelcateg.grid(pady=10, padx=25, row=0, column=1)

        self.subFr5.grid(row=3, column=1)

        self.subFr6 = tk.Frame(mainFrame)

        btn_save = ttk.Button(self.subFr6, text="Save grouping",
                              command=lambda: self.saveGrouping(controller), width=15)
        btn_save.grid(row=0, column=0, pady=10, columnspan=2)

        labelempty = tk.Label(self.subFr6, text="")
        labelempty.grid(padx=25, row=0, column=2)

        btn_fill = ttk.Button(self.subFr6, text="Fill remaining",
                              command=lambda: self.fillRemaining(controller), width=15)
        btn_fill.grid(row=0, column=3, pady=10, columnspan=2)

        self.subFr6.grid(row=4, column=1)

        self.subFr7 = tk.Frame(mainFrame)

        # TODO: unecessary repetition
        label = tk.Label(self.subFr7, text='Category', font=(controller._font, 10))
        label.grid(padx=10, row=0, column=0)

        label = tk.Label(self.subFr7, text='Bad cnt', font=(controller._font, 10))
        label.grid(padx=10, row=0, column=1)

        label = tk.Label(self.subFr7, text='Tot cnt', font=(controller._font, 10))
        label.grid(padx=10, row=0, column=2)

        label = tk.Label(self.subFr7, text='Def rx', font=(controller._font, 10))
        label.grid(padx=10, row=0, column=3)

        self.subFr7.grid(row=1, column=2, rowspan=3, sticky='n')

        if controller._col_cat_alias or controller._row_cat_alias:
            self.subFr8 = tk.Frame(mainFrame)
            label = tk.Label(self.subFr8, text='*Hover over category name to get list of group the category consists of.',
                             fg='red', font=(controller._font, "8", "normal"))
            label.grid(padx=10, row=0, column=0)
            self.subFr8.grid(row=5, column=1, columnspan=2, sticky='w')

    def selectCategory(self, btn_id, categ, controller):
        """
        Actions performed when individual category button is selected. (???)

        Args:
            btn_id ():
            categ ():
            controller ():
        """

        if categ in self.stats.keys():
            if self.btns[btn_id][4] == 1:
                old_cat = self.btns[btn_id][1]
                self.stats[old_cat][0] -= controller._bad_mtrx.iloc[self.btns[btn_id][2], self.btns[btn_id][3]]
                self.stats[old_cat][1] -= controller._cnt_mtrx.iloc[self.btns[btn_id][2], self.btns[btn_id][3]]
                self.stats[old_cat][2] = self.stats[categ][0] / self.stats[categ][1]

                # label = tk.Label(self.subFr7, text = str(old_cat))
                # label.grid(padx = 10, row = old_cat, column = 0)
                label = tk.Label(self.subFr7, text=str(int(self.stats[old_cat][0])), font=(controller._font, 10))
                label.grid(padx=10, row=old_cat, column=1)
                label = tk.Label(self.subFr7, text=str(int(self.stats[old_cat][1])), font=(controller._font, 10))
                label.grid(padx=10, row=old_cat, column=2)
                label = tk.Label(self.subFr7, text=str(round(self.stats[old_cat][2], 4)), font=(controller._font, 10))
                label.grid(padx=10, row=old_cat, column=3)

            self.stats[categ][0] += controller._bad_mtrx.iloc[self.btns[btn_id][2], self.btns[btn_id][3]]
            self.stats[categ][1] += controller._cnt_mtrx.iloc[self.btns[btn_id][2], self.btns[btn_id][3]]
            if self.stats[categ][1] > 0:
                self.stats[categ][2] = self.stats[categ][0] / self.stats[categ][1]
            else:
                self.stats[categ][2] = -1.0
        else:
            bad_cnt = controller._bad_mtrx.iloc[self.btns[btn_id][2], self.btns[btn_id][3]]
            tot_cnt = controller._cnt_mtrx.iloc[self.btns[btn_id][2], self.btns[btn_id][3]]
            if tot_cnt == 0:
                self.stats[categ] = [bad_cnt, tot_cnt, -1.0]
            else:
                self.stats[categ] = [bad_cnt, tot_cnt, bad_cnt / tot_cnt]

        label = tk.Label(self.subFr7, text=str(categ), font=(controller._font, 10))
        label.grid(padx=10, row=categ, column=0)
        label = tk.Label(self.subFr7, text=str(int(self.stats[categ][0])), font=(controller._font, 10))
        label.grid(padx=10, row=categ, column=1)
        label = tk.Label(self.subFr7, text=str(int(self.stats[categ][1])), font=(controller._font, 10))
        label.grid(padx=10, row=categ, column=2)
        label = tk.Label(self.subFr7, text=str(round(self.stats[categ][2], 4)), font=(controller._font, 10))
        label.grid(padx=10, row=categ, column=3)

        self.btns[btn_id][1] = categ
        self.btns[btn_id][0]['text'] = categ
        self.btns[btn_id][0].configure(fg='#%02x%02x%02x' % self.categ_colors[min(categ-1, 9)])

        self.btns[btn_id][4] = 1

    def previousCategory(self, controller):
        """
        Previous Category button control

        Args:
            controller ():
        """

        controller.categ -= 1
        controller.update()
        label = tk.Label(self.subFr5, text="Select category {}".format(controller.categ), font=(controller._font, 10))
        label.grid(pady=10, padx=10, row=0, column=1)

    def nextCategory(self, controller):
        """
        Next Category button control

        Args:
            controller ():
        """
        controller.categ += 1
        controller.show_frame(SelectCategories)
        label = tk.Label(self.subFr5, text="Select category {}".format(controller.categ), font=(controller._font, 10))
        label.grid(pady=10, padx=10, row=0, column=1)

    def fillRemaining(self, controller):
        """
        Fill Remaining button control
        TODO: does not work when no category was yet selected

        Args:
            controller ():
        """

        categ = max(self.stats.keys()) + 1

        for btn in self.btns:
            if self.btns[btn][4] == 0:
                self.btns[btn][4] = 1

                self.btns[btn][1] = categ
                self.btns[btn][0]['text'] = categ
                self.btns[btn][0].configure(fg='#%02x%02x%02x' % self.categ_colors[min(categ - 1, 9)])
                self.btns[btn][4] = 1

                if categ in self.stats.keys():
                    self.stats[categ][0] += controller._bad_mtrx.iloc[self.btns[btn][2], self.btns[btn][3]]
                    self.stats[categ][1] += controller._cnt_mtrx.iloc[self.btns[btn][2], self.btns[btn][3]]
                    self.stats[categ][2] = self.stats[categ][0] / self.stats[categ][1]
                else:
                    bad_cnt = controller._bad_mtrx.iloc[self.btns[btn][2], self.btns[btn][3]]
                    tot_cnt = controller._cnt_mtrx.iloc[self.btns[btn][2], self.btns[btn][3]]
                    self.stats[categ] = [bad_cnt, tot_cnt, bad_cnt / tot_cnt]

        if categ in self.stats.keys():
            label = tk.Label(self.subFr7, text=str(categ), font=(controller._font, 10))
            label.grid(padx=10, row=categ, column=0)
            label = tk.Label(self.subFr7, text=str(int(self.stats[categ][0])), font=(controller._font, 10))
            label.grid(padx=10, row=categ, column=1)
            label = tk.Label(self.subFr7, text=str(int(self.stats[categ][1])), font=(controller._font, 10))
            label.grid(padx=10, row=categ, column=2)
            label = tk.Label(self.subFr7, text=str(round(self.stats[categ][2], 4)), font=(controller._font, 10))
            label.grid(padx=10, row=categ, column=3)

    @staticmethod
    def popupmsg(msg):
        """
        Args:
            msg ():
        """

        popup = tk.Tk()
        popup.wm_title("Message")
        label = ttk.Label(popup, text=msg, font=("TkFixedFont", 10))
        label.pack(side="top", fill="x", pady=10)
        B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
        B1.pack()
        popup.mainloop()

    def saveGrouping(self, controller):
        """Function to save metadata from given matrix, created using tkInter.
        Gets all data from the GUI app.
        
        Args:
            controller (tk.Tk) tkinter instance which creates mapping
        """

        def process_intervals(numerical_cats):
            starts = [low for low, _ in numerical_cats]
            ends = [upp for _, upp in numerical_cats]
            return starts, ends
        time_start = tm.time()

        # boilerplate code
        inter_name = 'i_' + controller._row_var + '_' + controller._column_var
        controller.btns = self.btns
        controller.inter_name = inter_name
        controller._inter_coord[controller.inter_name] = np.zeros(len(controller._inter_coord)).astype(int)

        # not sure why this is needed - i guess some edge case solving
        if 'null' in controller._def_mtrx.index:
            controller._inter_coord[controller._row_var].replace(
                0, list(controller._def_mtrx.index).index('null')+1, inplace=True)
        # TODO: according to pandas docs, using inplace is discouraged practice and will be deprecated
        if 'null' in controller._def_mtrx.columns:
            controller._inter_coord[controller._column_var].replace(
                0, list(controller._def_mtrx.columns).index('null')+1, inplace=True)

        for btn in controller.btns:
            row = controller.btns[btn][2]+1
            col = controller.btns[btn][3]+1
            controller._inter_coord[controller.inter_name][(controller._inter_coord[controller._row_var] == row) & (
                controller._inter_coord[controller._column_var] == col)] = controller.btns[btn][1]
            controller.categories[row-1, col-1] = controller.btns[btn][1]

        # this is not nice.
        # in this part, we create the metadata. So for each category,
        # we create dictionary by which we can qualify the individual groups.
        metadata = {}
        for category in np.unique(controller.categories):

            chosen_category = np.where(controller.categories == category)

            # first case - both categories are categorical
            if controller._row_bins_numeric is None and controller._col_bins_numeric is None:
                categories_cols, categories_rows = np.meshgrid(controller._def_mtrx.columns, controller._def_mtrx.index)
                chosen_cols, chosen_rows = categories_cols[chosen_category], categories_rows[chosen_category]
                metadata[str(category)] = list(zip(chosen_cols, chosen_rows))

            # column variable is numerical, row variable categorical
            elif controller._row_bins_numeric is None:
                # dividing the zipped list to two lists
                starts_col, ends_col = process_intervals(controller._col_bins_numeric)
                # first meshgrid - creating table where i,j is exactly i, j - in this case interval start + category
                interval_starts_cols, _ = np.meshgrid(starts_col, controller._def_mtrx.index)
                # i, j with column interval end + category
                interval_ends_cols, categories_rows = np.meshgrid(ends_col, controller._def_mtrx.index)

                # zipping the column intervals back together
                chosen_cols = list(zip(interval_starts_cols[chosen_category],
                                       interval_ends_cols[chosen_category]))
                chosen_rows = categories_rows[chosen_category]

                # zipping columns with rows
                metadata[str(category)] = list(zip(chosen_cols,
                                                   chosen_rows))

            # column categorical, row numerical
            elif controller._col_bins_numeric is None:
                # this part is similar to the elif before - just switched the order
                # however decided not to make function out of this - the order of the variables is important
                starts_row, ends_row = process_intervals(controller._row_bins_numeric)
                _, interval_starts_rows = np.meshgrid(controller._def_mtrx.columns, starts_row)
                categories_cols, interval_ends_rows = np.meshgrid(controller._def_mtrx.columns, ends_row)
                chosen_cols = categories_cols[chosen_category]
                chosen_rows = list(zip(interval_starts_rows[chosen_category],
                                       interval_ends_rows[chosen_category]))
                metadata[str(category)] = list(zip(chosen_cols,
                                                   chosen_rows))
            
            # both variables are numerical
            else:
                starts_row, ends_row = process_intervals(controller._row_bins_numeric)
                starts_col, ends_col = process_intervals(controller._col_bins_numeric)

                # let the shitstorm begin.
                # First - creating interval starts for both column and row variable
                interval_starts_cols, interval_starts_rows = np.meshgrid(starts_col, starts_row)
                # Creating interval ends
                interval_ends_cols, interval_ends_rows = np.meshgrid(ends_col, ends_row)
                # So at this moment, we have 4 matrices.
                # 1) interval starts of column variable
                # 2) interval starts of row variable
                # 3) interval ends of column variable
                # 4) interval ends of row variable
                # these matrices are just matrices i, j. So now we can use the mask created at the top.
                # if somebody is going to rewrite this - I guess this is the way - just create different masks.
                # okay, we can go on - lets zip it all together.
                chosen_cols = list(zip(interval_starts_cols[chosen_category],
                                       interval_ends_cols[chosen_category]))
                chosen_rows = list(zip(interval_starts_rows[chosen_category],
                                       interval_ends_rows[chosen_category]))
                metadata[str(category)] = list(zip(chosen_cols,
                                                   chosen_rows))

        controller.metadata = metadata
        controller.categories = pd.DataFrame(controller.categories,
                                             columns = controller._def_mtrx.columns,
                                             index = controller._def_mtrx.index)
        self.popupmsg("Metadata will be saved.\nFeel free to close the window.\n\nSave duration: {:.2f} sec.".format(
            tm.time() - time_start))
        print('Done in ' + str(round(tm.time() - time_start, 2)) + ' sec.')


class CreateToolTip(object):
    '''
    create a tooltip for a given widget
    TODO: Could this class also be merged with InteractionsGUI?
    '''

    def __init__(self, widget, text='widget info'):

        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.close)

    def enter(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                         background='yellow', relief='solid', borderwidth=1,
                         font=('TkFixedFont', "10", "bold"))
        label.pack(ipadx=1)

    def close(self, event=None):
        if self.tw:
            self.tw.destroy()
