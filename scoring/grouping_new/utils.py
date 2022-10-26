from ipywidgets import widgets


class ChartsSelector:
    def __init__(self, parent=None):
        self.parent = parent

        self.widget = widgets.VBox(
            [
                widgets.Checkbox(value=True, description="Equifrequency fine classing", disabled=True),
                widgets.Checkbox(value=True, description="Equidistant fine classing", disabled=True),
                widgets.Checkbox(value=True, description="Final groups", disabled=True),
                widgets.Checkbox(value=True, description="Stability"),
            ],
            layout=widgets.Layout(width="auto"),
        )

        for checkbox in self.widget.children:
            checkbox.observe(self._choose_charts)

        self.selected_charts = [True for _ in self.widget.children]

    def _choose_charts(self, change):
        self.selected_charts = [b.value for b in self.widget.children]


class AutoGroupingTab:
    def __init__(self, parent=None):
        self.parent = parent

        self.w_group_auto_btn = widgets.Button(description="Group automaticaly")

        self.w_group_auto_btn.on_click(self.parent._automatic_grouping_one_column_from_gui)

        style = {"description_width": "initial"}
        self.w_group_count_text = widgets.IntText(
            # min=1,
            # max=10,
            value=5,
            layout=widgets.Layout(width="200px"),
            description="Max bins",
            style=style,
        )
        self.w_min_samples_num_text = widgets.IntText(
            # min=1,
            # max=1000,
            value=100,
            layout=widgets.Layout(min_width="200px"),
            description="Min group size",
            style=style,
        )

        self.w_min_samples_cat_text = widgets.IntText(
            # min=1,
            # max=1000000,
            value=100,
            layout=widgets.Layout(width="200px"),
            description="Min category size",
            style=style,
        )

        self.widget = widgets.VBox(
            [self.w_group_auto_btn, self.w_group_count_text, self.w_min_samples_num_text, self.w_min_samples_cat_text]
        )


class ToolsTab:
    def __init__(self, parent=None):
        self.parent = parent

        self.export_button = widgets.Button(description="Export charts")
        self.export_path = widgets.Text(
            value="documentation\\grouping", placeholder="Folder path", description="Export path:", disabled=False
        )

        self.export_button.on_click(self.parent._export_pictures)

        self.save_button = widgets.Button(description="Save grouping")
        self.save_path = widgets.Text(
            value="grouping_manual.json", placeholder="File path", description="Save path:", disabled=False
        )

        self.save_button.on_click(self.parent._save_manual_from_gui)

        self.widget = widgets.VBox(
            [
                widgets.HBox([self.export_button, self.export_path]),
                widgets.HBox([self.save_button, self.save_path]),
                # widgets.HBox(
                #     [
                #         widgets.Button(description="Load grouping"),
                #         widgets.FileUpload(
                #             accept=".json",  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
                #             multiple=False,  # True to accept multiple files upload else False
                #         ),
                #     ]
                # ),
            ]
        )


class BottomButtons:
    def __init__(self, parent=None):
        self.parent = parent

        self.close_button = widgets.Button(description="Save and close")
        self.close_button.on_click(self._close_all)

        buttons = [self.close_button]

        self.widget = widgets.HBox(buttons)

    def _close_all(self, btn):
        print("Grouping model saved.")
        self.parent._save_tab_changes()
        self.parent.widget.close()
